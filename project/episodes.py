import pandas as pd
import numpy as np
from shapely import Polygon
import geopandas as gpd


def get_region_follow_pairs_df(follow_pairs_df, polygon):
    points = gpd.GeoSeries(
        gpd.points_from_xy(
            follow_pairs_df["lat_follower"],
            follow_pairs_df["lon_follower"],
        )
    )
    region_follow_pairs_df = follow_pairs_df.loc[points.within(polygon)]
    return region_follow_pairs_df.copy()


def my_leader_is_in_subregion(region_follow_pairs_df, polygon):
    leader_points = gpd.GeoSeries(
        gpd.points_from_xy(
            region_follow_pairs_df["lat_leader"],
            region_follow_pairs_df["lon_leader"],
        ),
        index=region_follow_pairs_df.index,
    )
    return leader_points.within(polygon)


def find_horizontal_line_with_min_intersections(series):
    min_val = int(series.quantile(0.2))
    max_val = int(series.quantile(0.5))
    if min_val == 0:
        min_val = 1

    def count_crossings(values, threshold):
        above_threshold = values > threshold
        crossings = np.diff(above_threshold.astype(int))
        total_crossings = np.sum(np.abs(crossings))
        return total_crossings

    search_space = np.arange(min_val, max_val, 1)
    num_crossings = [count_crossings(series, thres) for thres in search_space]
    return search_space[np.array(num_crossings).argmin()]


def filter_green_lights_by_length(time_periods, min_duration=20):
    series = pd.Series(time_periods["speed_follower"])
    episode_groups = (series != series.shift()).cumsum()
    episode_lengths = series.groupby(episode_groups).transform("sum")
    filtered_series = series & (episode_lengths >= min_duration / 0.04)
    return filtered_series


def get_region_green_lights(region_follow_pairs_df):
    region_mean_speed_in_area_per_timestep = region_follow_pairs_df.groupby(
        "time_follower"
    )[["speed_follower"]].mean()
    thres = find_horizontal_line_with_min_intersections(
        region_mean_speed_in_area_per_timestep["speed_follower"]
    )

    # True when green
    time_periods = region_mean_speed_in_area_per_timestep >= thres

    # green duration more than 20 sec
    green_lights = filter_green_lights_by_length(time_periods, min_duration=20)
    return green_lights


def get_region_global_flow_reward_per_timestep(
    region_follow_pairs_df, region_green_lights
):
    count_vehicles_per_timestep = region_follow_pairs_df.groupby("time_follower")[
        "track_id_follower"
    ].count()
    vehicles_diff_per_timestep = count_vehicles_per_timestep.diff().bfill()
    # TODO: rethink this
    reward_per_timestep = -(vehicles_diff_per_timestep != 0).astype(int)
    return reward_per_timestep


def episodes_from_follow_pairs_df(follow_pairs_df, subregions):
    episodes = []
    for region_name, polygon in subregions.items():
        r_follow_pairs_df = get_region_follow_pairs_df(follow_pairs_df, polygon)
        r_green_lights = get_region_green_lights(r_follow_pairs_df)
        global_reward = get_region_global_flow_reward_per_timestep(
            r_follow_pairs_df, r_green_lights
        )
        r_follow_pairs_df.loc[:, "green_light"] = r_follow_pairs_df[
            "time_follower"
        ].map(r_green_lights)
        r_follow_pairs_df.loc[:, "flow_reward"] = r_follow_pairs_df[
            "time_follower"
        ].map(global_reward)
        r_follow_pairs_df.loc[:, "episode_id"] = r_follow_pairs_df.groupby(
            "track_id_follower"
        )["track_id_follower"].transform(lambda g: f"{region_name}_{g.name}")
        # careful: if follower doesn't have a leader it will show False
        r_follow_pairs_df.loc[:, "leader_in_followers_subregion"] = (
            my_leader_is_in_subregion(r_follow_pairs_df, polygon)
        )
        episodes.append(r_follow_pairs_df)

    # concat
    episodes_df = pd.concat(episodes, axis=0, ignore_index=True)

    # filter episodes more than 128 transitions
    long_enough_episodes = (
        episodes_df.groupby("episode_id")["track_id_follower"].count() >= 128
    )
    filtered_episode_ids = long_enough_episodes[long_enough_episodes].index
    episodes_df = episodes_df.loc[episodes_df["episode_id"].isin(filtered_episode_ids)]

    return episodes_df


subregions = {
    "d2_r1": Polygon(
        [  # panepistimiou: voukourestiou - amerikis
            (37.97738381991755, 23.735682737010652),
            (37.978111122155276, 23.73507119337037),
            (37.97789969782921, 23.73474932829654),
            (37.97722313589298, 23.735350143101027),
        ]
    ),
    "d2_r2": Polygon(
        [  # panepistimiou: amerikis - omirou
            (37.97826334726513, 23.734963905027538),
            (37.978711563892446, 23.734534751595756),
            (37.978550882774776, 23.73422361535772),
            (37.978077294276225, 23.7346420399537),
        ]
    ),
    "d2_r3": Polygon(
        [  # panepistimiou: omirou - sina
            (37.97885533091011, 23.73441673440202),
            (37.97936274166243, 23.73405195398501),
            (37.97917669146058, 23.73370863123959),
            (37.978694650107165, 23.73412705583557),
        ]
    ),
    "d2_r4": Polygon(
        [  # panepistimiou: sina - panepistimio
            (37.97951496423183, 23.733891021432985),
            (37.98033526915966, 23.733140002927374),
            (37.980149221423304, 23.732839595525128),
            (37.9793373712359, 23.73359061403074),
        ]
    ),
    "d2_r5": Polygon(
        [  # stadiou: amerikis - omirou
            (37.97758248134508, 23.733790521126636),
            (37.978064530003266, 23.73336136769486),
            (37.97800110272913, 23.733136062143178),
            (37.977468311462324, 23.733554486739163),
        ]
    ),
    "d2_r6": Polygon(
        [  # stadiou: omirou - sina
            (37.97817024200499, 23.733291630262197),
            (37.97869034283615, 23.732841019158833),
            (37.97858040313028, 23.732621078025044),
            (37.978072986969, 23.733060960292615),
        ]
    ),
    "d2_r7": Polygon(
        [  # stadiou: sina - klathmonos
            (37.978785439008256, 23.732750278527696),
            (37.97926325135144, 23.732347947185403),
            (37.97915331250376, 23.73214946372321),
            (37.9786628141, 23.732541066229704),
        ]
    ),
    "d3_r1": Polygon(
        [  # panepistimiou: panepistimio - ippokratous
            (37.980622445384775, 23.732855244296623),
            (37.9811101953576, 23.732466100656815),
            (37.98096940188381, 23.73218540688384),
            (37.9804514807095, 23.732612826947232),
        ]
    ),
    "d3_r2": Polygon(
        [  # panepistimiou: ippokratous - pesmazogloy
            (37.98120903162378, 23.732353026167658),
            (37.98157689297063, 23.73205261876541),
            (37.98146272930171, 23.731795126706345),
            (37.9811033239991, 23.732068712019103),
        ]
    ),
    "d3_r3": Polygon(
        [  # panepistimiou: pezmazoglou - trikoupi
            (37.98157689298724, 23.732041889910942),
            (37.98201240461169, 23.73166101624024),
            (37.98191515466726, 23.73139279534538),
            (37.981462729318295, 23.731746846926594),
        ]
    ),
    "d3_r4": Polygon(
        [  # panepistimiou: trikoupi - mpenaki
            (37.98212974513109, 23.731574486362867),
            (37.983229081607966, 23.73063571323085),
            (37.98309800773897, 23.730378221171787),
            (37.98201135406664, 23.73135454522908),
        ]
    ),
    "d3_r5": Polygon(
        [  # stadiou: klathmonos - pesmazogloy
            (37.980202864791075, 23.73156048644483),
            (37.980904769183226, 23.731002586980996),
            (37.98078214781609, 23.730771917011413),
            (37.980097155720216, 23.73135663856221),
        ]
    ),
    "d3_r6": Polygon(
        [  # stadiou: pezmazoglou - arsaki
            (37.98102132017514, 23.730910056279548),
            (37.981452606808965, 23.730566733534125),
            (37.981342671241045, 23.73032533472875),
            (37.98091561228009, 23.73071157281735),
        ]
    ),
    "d3_r7": Polygon(
        [  # stadiou: arsaki - mpenaki
            (37.981763741181304, 23.73027323062894),
            (37.982634759634635, 23.72954903420896),
            (37.9825544234146, 23.72931836423938),
            (37.98173414329235, 23.72999428089443),
        ]
    ),
}

for region in ["d2", "d3"]:
    follow_pairs_df = pd.read_csv(
        f"/project/datasets/follow_pairs_20181029_{region}_1000_1030.csv",
    )
    follow_pairs_df["leader_follower_dist"] = follow_pairs_df[
        "leader_follower_dist"
    ].apply(lambda x: float(x[1:-1]) if not isinstance(x, float) else np.nan)

    episodes_df = episodes_from_follow_pairs_df(
        follow_pairs_df,
        {
            subregion_name: polygon
            for subregion_name, polygon in subregions.items()
            if region in subregion_name
        },
    )

    episodes_df.to_csv(
        f"/project/datasets/episodes_20181029_{region}_1000_1030.csv", index=False
    )
