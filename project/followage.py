import pandas as pd
import numpy as np
from shapely import LineString
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


def preprocess(df: pd.DataFrame):
    df["delta_lat"] = -df.groupby("track_id")["lat"].diff(periods=-1)
    df["delta_lon"] = -df.groupby("track_id")["lon"].diff(periods=-1)

    df["displ_angle"] = np.arctan2(df["delta_lat"], df["delta_lon"])
    df["lon_vel"] = df["speed"] * np.cos(df["displ_angle"])
    df["lat_vel"] = df["speed"] * np.sin(df["displ_angle"])

    # exclude (vehicle, time) pairs that are stopped
    # df = df.loc[~((df['speed'].abs() < 1e-5) & (df['lon_acc'].abs() < 1e-5) & (df['lat_acc'].abs() < 1e-5))]

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

    # subsample time to 200ms
    # df = df.loc[(df["time"] * 100).astype(int) % 8 == 0]

    df.sort_values(by=["track_id", "time"], inplace=True)

    # drop transitions that don't move till end
    df[["next_lat", "next_lon"]] = df.groupby("track_id")[["lat", "lon"]].shift(-1)
    df["lat_lon_changed"] = (df["lat"] != df["next_lat"]) | (
        df["lon"] != df["next_lon"]
    )
    df.loc[~df["lat_lon_changed"], "next_lat"] = np.nan
    df.loc[~df["lat_lon_changed"], "next_lon"] = np.nan
    df["next_lat"] = df.groupby("track_id")["next_lat"].bfill()
    df["next_lon"] = df.groupby("track_id")["next_lon"].bfill()
    df = df.loc[(df["next_lon"].notna()) & (df["next_lat"].notna())]

    df["time_index"] = df.groupby("track_id").cumcount()

    df["points"] = df[["lat", "lon"]].apply(lambda x: (x["lat"], x["lon"]), axis=1)
    df["lpoints"] = df["points"].apply(lambda x: [x])
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

        # left has right around him
        left_area_around = left_dil_traj_gdf.intersection(
            left_points.buffer(AROUND_BUFFER)
        )
        left_has_right_around = left_area_around.contains(right_points)

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
        follow: pd.DataFrame = cross_df.loc[left_follows_right][
            ["track_id_left", "track_id_right", "time_left"]
        ]

        # if left follows multiple right, keep closest
        fleft = left_points.loc[follow.index]
        fright = right_points.loc[follow.index]
        follow["dist"] = fleft.distance(fright)
        follow = follow.loc[
            follow["dist"] == follow.groupby("track_id_left")["dist"].transform("min")
        ]

        # append
        followage.append(follow[["track_id_left", "track_id_right", "time_left"]])

    return pd.concat(followage, ignore_index=True)


df = csv_to_df("/project/datasets/20181029_d1_1000_1030.csv")
df = preprocess(df)


_, AROUND_BUFFER = lat_lon_buffer(df["lat"].mean(), 30)
_, TRAJECTORY_BUFFER = lat_lon_buffer(df["lat"].mean(), 2)

dil_traj_gdf = gpd.GeoSeries(
    df.groupby("track_id")["lpoints"].sum().apply(LineString)
).buffer(TRAJECTORY_BUFFER)

follow_df = find_followage(df)

follow_df.to_csv("/project/datasets/followage_df_20181029_d1_1000_1030.csv")
df.to_csv("/project/datasets/df_20181029_d1_1000_1030.csv")
