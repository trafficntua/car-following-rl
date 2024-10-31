import pandas as pd
import numpy as np
from shapely import LineString, Point
import geopandas as gpd
from tqdm import tqdm


def csv_to_df(file_path):
    raw_df = pd.read_csv(file_path, header=None, skiprows=1)
    result = []
    for _, row in raw_df.iterrows():
        row_data = row.str.split(";").to_list()[0]
        track_id, vehicle_type, traveled_d, avg_speed, *the_rest = row_data

        for i in range(0, len(the_rest), 6):
            if the_rest[i] == " ":
                break

            result.append(
                {
                    "track_id": int(track_id),
                    "vehicle_type": vehicle_type,
                    "traveled_d": np.float64(traveled_d),
                    "avg_speed": np.float64(avg_speed),
                    "lat": np.float64(the_rest[i]),
                    "lon": np.float64(the_rest[i + 1]),
                    "speed": np.float64(the_rest[i + 2]),
                    "lon_acc": np.float64(the_rest[i + 3]),
                    "lat_acc": np.float64(the_rest[i + 4]),
                    "time": np.float64(the_rest[i + 5]),
                }
            )
    final_df = pd.DataFrame(result)
    final_df.sort_values(by=["track_id", "time"], inplace=True)
    return final_df


def haversine(lon1: float, lat1: float, lon2: float, lat2: float):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def preprocess(df: pd.DataFrame):
    # lat, lon speed
    df["prev_lat"] = df.groupby("track_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("track_id")["lon"].shift(1)

    df["curr_prev_lat_diff_m"] = (df["lat"] - df["prev_lat"]) * 111320
    df["curr_prev_lon_diff_m"] = (
        (df["lon"] - df["prev_lon"]) * 111320 * np.cos(np.radians(df["lat"]))
    )

    df["speed_lat"] = (
        (df["curr_prev_lat_diff_m"] / 0.04 * 3.6).rolling(11, center=True).mean()
    )
    df["speed_lon"] = (
        (df["curr_prev_lon_diff_m"] / 0.04 * 3.6).rolling(11, center=True).mean()
    )
    df["speed2"] = np.sqrt(df["speed_lat"] ** 2 + df["speed_lon"] ** 2)

    # drop temp cols
    df.drop(columns=["curr_prev_lat_diff_m", "curr_prev_lon_diff_m"], inplace=True)

    # drop transitions in the begining and end of trajectory
    df = df.loc[(df["speed_lat"].notna()) & (df["speed_lon"].notna())]

    # exclude Motorcycle and Taxis
    df = df.loc[(df["vehicle_type"] != " Motorcycle") & (df["vehicle_type"] != " Taxi")]

    # exclude vehicles that are stopped > 85% of their time
    exclude_vehicles_85 = set()
    for (track_id,), df_id in df.groupby(by=["track_id"]):
        stopped_timesteps = len(
            df_id[
                (df_id["speed"].abs() < 1e-5)
                & (df_id["lon_acc"].abs() < 1e-5)
                & (df_id["lat_acc"].abs() < 1e-5)
            ]
        )
        total_timesteps = len(df_id)
        stopped_percent = stopped_timesteps / total_timesteps
        if stopped_percent > 0.85:
            exclude_vehicles_85.add(track_id)
    df = df.loc[(~df["track_id"].isin(exclude_vehicles_85))]

    df.sort_values(by=["track_id", "time"], inplace=True)

    # if position don't change in a step, get next position that does change
    df[["next_lat", "next_lon"]] = df.groupby("track_id")[["lat", "lon"]].shift(-1)
    df["lat_lon_changed"] = (df["lat"] != df["next_lat"]) | (
        df["lon"] != df["next_lon"]
    )
    df.loc[~df["lat_lon_changed"], "next_lat"] = np.nan
    df.loc[~df["lat_lon_changed"], "next_lon"] = np.nan
    df["next_lat"] = df.groupby("track_id")["next_lat"].bfill()
    df["next_lon"] = df.groupby("track_id")["next_lon"].bfill()

    # drop transitions that don't move till end
    df = df.loc[(df["next_lon"].notna()) & (df["next_lat"].notna())]

    # drop track_ids that have only one transition
    df = df.loc[df.groupby("track_id")["lat"].transform("count") > 1]

    df["time_index"] = df.groupby("track_id").cumcount()

    df["points"] = df[["lat", "lon"]].apply(lambda x: (x["lat"], x["lon"]), axis=1)
    df["lpoints"] = df["points"].apply(lambda x: [x])

    df["step_dist"] = 1000 * df.apply(
        lambda row: haversine(row["lon"], row["lat"], row["prev_lon"], row["prev_lat"]),
        axis=1,
    )
    df["cum_dist"] = df.groupby("track_id")["step_dist"].cumsum()
    return df


def lat_lon_buffer(lat: float, in_meters: float):
    """
    If your displacements aren't too great (less than a few kilometers)
    and you're not right at the poles, use the quick and dirty estimate
    that 111,111 meters (111.111 km) in the y direction is 1 degree (of latitude)
    and 111,111 * cos(latitude) meters in the x direction is 1 degree (of longitude).

    Length in km of 1° of latitude = always 111.32 km
    Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360
    """
    METERS_PER_DEGREE_LAT: float = 111320
    METERS_PER_DEGREE_LON: float = 111320 * np.cos(np.radians(lat))

    buffer_size_lat = in_meters / METERS_PER_DEGREE_LAT
    buffer_size_lon = in_meters / METERS_PER_DEGREE_LON
    return buffer_size_lat, buffer_size_lon


def find_followage(df: pd.DataFrame):
    """
    For every time step find, for every vehicle, find the vehicle that leads.
    Gather the results in a new dataframe
    """
    followage = []
    for _, df_time in tqdm(df.groupby(by=["time"])):
        # cross join and check if left follows right
        cross_df = df_time.merge(df_time, how="cross", suffixes=("_left", "_right"))
        left_dil_traj_gdf = gpd.GeoSeries(cross_df["track_id_left"].map(dil_traj_gdf))

        # points
        left_points = gpd.GeoSeries(
            gpd.points_from_xy(cross_df["lat_left"], cross_df["lon_left"]),
            index=cross_df.index,
        )
        right_points = gpd.GeoSeries(
            gpd.points_from_xy(cross_df["lat_right"], cross_df["lon_right"]),
            index=cross_df.index,
        )

        # left is not right
        left_not_equal_right = cross_df["track_id_left"] != cross_df["track_id_right"]

        # left has right in his trajectory
        left_has_right_around = left_dil_traj_gdf.contains(right_points)

        # left moves towards right
        left_next_displacement = (
            cross_df[["next_lat_left", "next_lon_left"]].values
            - cross_df[["lat_left", "lon_left"]].values
        )
        left_to_right_displacement = (
            cross_df[["lat_right", "lon_right"]].values
            - cross_df[["lat_left", "lon_left"]].values
        )
        inner_prod = (left_next_displacement * left_to_right_displacement).sum(axis=1)
        left_moves_towards_right = inner_prod > 0

        # left follows right
        left_follows_right = (
            left_has_right_around & left_not_equal_right & left_moves_towards_right
        )
        follow = cross_df.loc[left_follows_right][
            ["track_id_left", "track_id_right", "time_left"]
        ]

        # if left follows multiple right, keep closest
        fleft = left_points.loc[follow.index]
        fright = right_points.loc[follow.index]
        follow.loc[:, "dist"] = fleft.distance(fright)
        follow = follow.loc[
            follow["dist"] == follow.groupby("track_id_left")["dist"].transform("min")
        ]

        # append
        followage.append(follow[["track_id_left", "track_id_right", "time_left"]])

    return pd.concat(followage, ignore_index=True)


def find_distance_between_follower_leader(follow_pairs_df: pd.DataFrame):
    fast_hash_map = follow_pairs_df[
        ["track_id_follower", "time_index_follower", "cum_dist_follower"]
    ].set_index(["track_id_follower", "time_index_follower"])

    def find_followers_closesest_trajectory_point_to_leader(row):
        followers_points_gdf = traj_points_gdf.loc[row["track_id_follower"]]
        distances = followers_points_gdf.distance(
            Point(row[["lat_leader", "lon_leader"]])
        )
        min_dist_follower_traj_point_idx = distances.idxmin()
        current_follower_cum_dist = row["cum_dist_follower"]
        current_follower_step_dist = row["step_dist_follower"]
        future_follower_cum_dist = fast_hash_map.loc[
            (row["track_id_follower"], min_dist_follower_traj_point_idx)
        ]
        dist = (
            future_follower_cum_dist
            - current_follower_cum_dist
            + current_follower_step_dist
        ).values[0]
        return dist

    follow_pairs_df["leader_follower_dist"] = follow_pairs_df.loc[
        ~follow_pairs_df["track_id_leader"].isna()
    ].apply(find_followers_closesest_trajectory_point_to_leader, axis=1)

    # drop false positive followage
    leader_cols = follow_pairs_df.columns[
        follow_pairs_df.columns.str.contains("leader")
    ]
    follow_pairs_df.loc[(follow_pairs_df["leader_follower_dist"] < 0), leader_cols] = (
        np.nan
    )


df = csv_to_df("/project/datasets/20181029_d1_1000_1030.csv")
df = preprocess(df)

# Dilation radius around 2 meters
_, TRAJECTORY_BUFFER = lat_lon_buffer(df["lat"].mean(), 2)

# trajectory Points, indexed by track_id, time_index
traj_points_gdf = gpd.GeoSeries(
    df.groupby("track_id")[["lat", "lon"]]
    .apply(lambda g: g.apply(lambda r: Point(r), axis=1).rename("point"))
    .reset_index()
    .set_index(["track_id", df.groupby("track_id").cumcount()])
    .rename_axis(["track_id", "time_index"])["point"]
)

# dilated trajectory Polygons, indexed by track_id
dil_traj_gdf = gpd.GeoSeries(
    df.groupby("track_id")["lpoints"].sum().apply(LineString)
).buffer(TRAJECTORY_BUFFER)

# followage
follow_df = find_followage(df)

# df join follow_df join df
follow_pairs_df = df.merge(
    follow_df,
    how="left",
    left_on=["track_id", "time"],
    right_on=["track_id_left", "time_left"],
).merge(
    df,
    how="left",
    left_on=["track_id_right", "time_left"],
    right_on=["track_id", "time"],
    suffixes=("_follower", "_leader"),
)

# follower-leader and clean false positive followages
find_distance_between_follower_leader(follow_pairs_df)

# drop useless columns
follow_pairs_df.drop(
    columns=["points_follower", "lpoints_follower", "points_leader", "lpoints_leader"],
    inplace=True,
)

# to csv
follow_pairs_df.to_csv(
    "/project/datasets/follow_pairs_20181029_d1_1000_1030.csv", index=False
)
