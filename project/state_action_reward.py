import pandas as pd
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm


sys.path.append("/usr/share/sumo/tools")
import traci

traci.start(["sumo", "-c", "./simulation2/basic_network.sumocfg"])
traci.vehicle.add(
    "follower",
    routeID="r1",
    departLane="first",
    typeID='Car',
    departPos=100,
)
traci.simulationStep()


def compute_vsafe_on_dataset(
    follower_type: str,
    leader_type: str,
    bumper_to_bumper_gap,
    follower_speed,
    leader_speed,
):
    vehicle_type_decels = {
        "Car": 4.6751,
        "Taxi": 4.7778,
        "Bus": 2.3356,
        "MediumVehicle": 7.5270,
        "HeavyVehicle": 5.7853,
    }
    clean_follower_type = follower_type.replace(' ', '')
    clean_leader_type = leader_type.replace(' ', '')
    # leader_max_comfortable_decel
    leader_max_comfortable_decel = vehicle_type_decels[clean_leader_type]

    # follow speed
    traci.vehicle.setType('follower', clean_follower_type)

    vsafe = traci.vehicle.getFollowSpeed(
        "follower",
        follower_speed,
        bumper_to_bumper_gap,
        leader_speed,
        leader_max_comfortable_decel,
    )
    return vsafe

# test 
print(compute_vsafe_on_dataset("Car", "MediumVehicle", 10, 20, 20))
print(compute_vsafe_on_dataset("Car", "Bus", 10, 20, 20))

episodes_d2 = pd.read_csv(
    "/project/datasets/episodes_20181029_d2_1000_1030.csv",
)

episodes_d3 = pd.read_csv(
    "/project/datasets/episodes_20181029_d3_1000_1030.csv",
)

episodes_df = pd.concat([episodes_d2, episodes_d3], ignore_index=True)

# keep episodes with leader
leader_set_per_episode = episodes_df.groupby("episode_id")["track_id_leader"].apply(
    lambda x: x.isna().any()
)
episodes_df = episodes_df.loc[
    episodes_df["episode_id"].isin(
        leader_set_per_episode[~leader_set_per_episode].index
    )
]

# m/s
episodes_df.loc[:, "speed_follower_m_s"] = episodes_df["speed_follower"] / 3.6
episodes_df.loc[:, "speed_leader_m_s"] = episodes_df["speed_leader"] / 3.6

# vehicle type stats
vehicle_lengths = pd.Series(
    [5, 5, 12.5, 5.83, 12.5, 2.5],
    index=[" Car", " Taxi", " Bus", " Medium Vehicle", " Heavy Vehicle", " Motorcycle"],
    name="length",
)
vehicle_max_speed = episodes_df.groupby("vehicle_type_follower")[
    "speed_follower_m_s"
].max()
print(vehicle_max_speed)

vehicle_max_accel = episodes_df.groupby("vehicle_type_follower")[
    "lon_acc_follower"
].apply(lambda x: x[x > 0].max())
print(vehicle_max_accel)

vehicle_max_decel = episodes_df.groupby("vehicle_type_follower")[
    "lon_acc_follower"
].apply(lambda x: x[x < 0].abs().max())
print(vehicle_max_decel)

vehicle_mean_decel = episodes_df.groupby("vehicle_type_follower")[
    "lon_acc_follower"
].apply(lambda x: x[x < 0].abs().mean())
print(vehicle_mean_decel)

# gap
gap = (
    episodes_df["leader_follower_dist"]
    - episodes_df["vehicle_type_follower"].map(vehicle_lengths) / 2
    - episodes_df["vehicle_type_leader"].map(vehicle_lengths) / 2
)
gap[gap < 0] = 0.1
episodes_df.loc[:, "leader_follower_gap"] = gap

# local reward
episodes_df["v_safe"] = episodes_df.apply(
    lambda row: compute_vsafe_on_dataset(
        row['vehicle_type_follower'][1:],
        row['vehicle_type_leader'][1:],
        row['leader_follower_gap'],
        row['speed_follower_m_s'],
        row['speed_leader_m_s'],
    ),
    axis=1
)

traci.close()

# local reward: triangular around vsafe
sigma = 0.2
episodes_df.loc[episodes_df['speed_leader_m_s'] < episodes_df['v_safe'], 'local_reward'] = (1 / sigma / episodes_df['v_safe']) * episodes_df['speed_leader_m_s'] - 1 / sigma
episodes_df.loc[episodes_df['speed_leader_m_s'] >= episodes_df['v_safe'], 'local_reward'] = - (1 / sigma / episodes_df['v_safe']) * episodes_df['speed_leader_m_s'] + 1 / sigma
episodes_df.loc[episodes_df['speed_leader_m_s'] <= (1 - sigma) * episodes_df['v_safe'], 'local_reward'] = -1
episodes_df.loc[episodes_df['speed_leader_m_s'] >= (1 + sigma) * episodes_df['v_safe'], 'local_reward'] = -1
episodes_df['local_reward'] = episodes_df['local_reward'] * (episodes_df['green_light'] | episodes_df['leader_in_followers_subregion']).astype(int)

# total reward
episodes_df["total_reward"] = episodes_df["local_reward"]

# dataset
observations = episodes_df[
    [
        "episode_id",
        "leader_follower_gap",
        "speed_follower_m_s",
        "speed_leader_m_s",
    ]
]

observations_dt = (
    observations.groupby("episode_id")[
        ["leader_follower_gap", "speed_follower_m_s", "speed_leader_m_s"]
    ]
    .apply(lambda g: g.values.tolist())
    .to_list()
)


action = episodes_df[["episode_id", "lon_acc_follower"]]
action_dt = action.groupby("episode_id")["lon_acc_follower"].apply(list).to_list()

reward = episodes_df[["episode_id", "total_reward"]]
reward_dt = reward.groupby("episode_id")["total_reward"].apply(list).to_list()

dones_dt = [[False] * len(action_dt[i]) for i in range(len(action_dt))]

train_dataset = [
    {
        "observations": ep_obs,
        "actions": [[act] for act in ep_act],
        "rewards": ep_rew,
        "dones": ep_done,
    }
    for ep_obs, ep_act, ep_rew, ep_done in zip(
        observations_dt, action_dt, reward_dt, dones_dt
    )
]

pickle.dump(
    train_dataset,
    open(
        "/project/datasets/train_dataset_20181029_d23_1000_1030.pkl",
        "wb",
    ),
)
