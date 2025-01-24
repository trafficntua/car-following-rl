import sys
from typing import Any
import pickle

sys.path.append("/usr/share/sumo/tools")
#from model.predict_action import predict # Comment SINGLE
from model.batch_predict_action import batch_predict
import traci
import pandas as pd

SIGMA = 0.2
RADIUS_NEIGHBOUR = 100
DO_CONTROL = False
SIMU_MINS = 5
MAX_EPISODE_LENGTH = 4095

def warmup():
    
    previous_vehicle_set = set()
    global_vehicles_exited = set()
    global_vehicles_entered = set()
    while len(global_vehicles_exited) < 100:
        traci.simulationStep()
        current_vehicle_set = set(traci.vehicle.getIDList())
        vehicles_entered = current_vehicle_set - previous_vehicle_set
        vehicles_exited = previous_vehicle_set - current_vehicle_set

        if vehicles_entered:
            global_vehicles_entered.update(vehicles_entered)

        if vehicles_exited:
            global_vehicles_exited.update(vehicles_exited)
        previous_vehicle_set = current_vehicle_set
    return current_vehicle_set


def state_neighbour(vehicle_ids: list[str]):
    positions = {
        follower_id: traci.vehicle.getPosition(follower_id)
        for follower_id in vehicle_ids
    }
    speeds = {
        follower_id: traci.vehicle.getSpeed(follower_id) for follower_id in vehicle_ids
    }

    df_vehicles = pd.DataFrame.from_dict(positions, orient="index", columns=["x", "y"])
    df_vehicles["speed"] = pd.Series(speeds)

    df_cross = df_vehicles.reset_index().merge(
        df_vehicles.reset_index(), how="cross", suffixes=("_ego", "_nbr")
    )

    df_cross["distance"] = (
        (df_cross["x_ego"] - df_cross["x_nbr"]) ** 2
        + (df_cross["y_ego"] - df_cross["y_nbr"]) ** 2
    ) ** 0.5

    df_neighbors = df_cross[df_cross["distance"] <= RADIUS_NEIGHBOUR]

    stats = (
        df_neighbors.groupby("index_ego")
        .agg(
            count_radius=("index_nbr", "count"),
            mean_speed_radius=("speed_nbr", "mean"),
            std_speed_radius=("speed_nbr", "std"),
            mean_dist_radius=("distance", "mean"),
            std_dist_radius=("distance", "std"),
        )
        .fillna(0)
    )

    return stats.to_dict("index")


def state_follower(vehicle_ids: list[str]):
    """State has the vehicle ids that we actually control as keys"""
    state = {}
    all_trafic_light_states = set()
    for follower_id in vehicle_ids:
        follower_speed = traci.vehicle.getSpeed(follower_id)
        leader = traci.vehicle.getLeader(follower_id, 500.0)
        upcoming_tls = traci.vehicle.getNextTLS(follower_id)

        for tls in upcoming_tls:
            all_trafic_light_states.add(tls[3])

        assert all_trafic_light_states <= {"G", "g", "y", "r"}, all_trafic_light_states

        green_light = (
            upcoming_tls[0][3] in {"G", "g", "y"}
            if len(upcoming_tls) != 0 and len(upcoming_tls[0]) == 4
            else False
        )
        follower_accel = traci.vehicle.getAcceleration(follower_id)
        follower_edge = traci.vehicle.getRoadID(follower_id)

        if leader is None:
            continue

        leader_id, gap_minus_min_gap = leader
        leader_edge = traci.vehicle.getRoadID(leader_id)
        min_gap = traci.vehicle.getMinGap(follower_id)
        gap = gap_minus_min_gap + min_gap
        leader_speed = traci.vehicle.getSpeed(leader_id)
        leader_max_comfortable_decel = traci.vehicle.getDecel(leader_id)
        kraus_follow_speed = traci.vehicle.getFollowSpeed(
            follower_id,
            follower_speed,
            gap + min_gap,
            leader_speed,
            leader_max_comfortable_decel,
        )
        leader_in_followers_subregion = follower_edge == leader_edge
        local_reward = get_local_reward(follower_speed, kraus_follow_speed) * (
            green_light or leader_in_followers_subregion
        )

        state[follower_id] = {
            "gap": gap,
            "follower_speed": follower_speed,
            "leader_speed": leader_speed,
            "green_light": +green_light,
            "local_reward": local_reward,
            "follower_accel": follower_accel,
        }
    return state


def get_flow_reward(
    prev_edge_vehicle_count: dict[str, int],
    curr_edge_vehicle_count: dict[str, int],
) -> dict[str, int]:
    assert prev_edge_vehicle_count.keys() == curr_edge_vehicle_count.keys()
    flow_rewards = {}
    for edge_id, _ in curr_edge_vehicle_count.items():
        diff = curr_edge_vehicle_count[edge_id] - prev_edge_vehicle_count[edge_id]
        flow_rewards[edge_id] = 1 if diff < 0 else 0
    return flow_rewards


def get_local_reward(follower_speed, v_safe):
    if follower_speed <= (1 - SIGMA) * v_safe:
        return -1
    if follower_speed <= v_safe:
        return (1 / SIGMA / v_safe) * follower_speed - 1 / SIGMA
    if follower_speed < (1 + SIGMA) * v_safe:
        return -(1 / SIGMA / v_safe) * follower_speed + 1 / SIGMA
    return -1


def simulate(ignore_vehicle_ids: set[str]):
    episode_stats = []

    # keys are episodes, values are lists of state-action-rewards
    trajectories: dict[str, Any] = {}

    # episode is the lifetime of a vehicle inside an edge
    episodes_done = set()

    # for every edge: keep how many vehicles it has
    prev_edge_vehicle_count = {
        edge_id: len(traci.edge.getLastStepVehicleIDs(edge_id))
        for edge_id in traci.edge.getIDList()
    }
    init_time = traci.simulation.getTime()
    print(f'Start control at time: {init_time} with {len(traci.vehicle.getIDList())} vehicles in the network(ignored)')
    curr_time = init_time

    previous_vehicle_set = set()
    global_vehicles_exited = set()
    global_vehicles_entered = set()

    while curr_time - init_time < 60 * SIMU_MINS:
        if (curr_time - init_time) % 60 == 0: 
            print(f'Time: {curr_time}, time elapsed: {(curr_time - init_time)/60} mins, vehs in network: {len(previous_vehicle_set)}')
        # run one step
        traci.simulationStep()

        # all vehicles currently in the simulation
        vehicle_id_list = list(traci.vehicle.getIDList())

        # keep track of vehicles in-out
        current_vehicle_set = set(vehicle_id_list) - ignore_vehicle_ids
        vehicles_entered = current_vehicle_set - previous_vehicle_set
        vehicles_exited = previous_vehicle_set - current_vehicle_set

        if vehicles_entered:
            global_vehicles_entered.update(vehicles_entered)

        if vehicles_exited:
            global_vehicles_exited.update(vehicles_exited)

        # no lane changes for all vehicles
        [
            traci.vehicle.setLaneChangeMode(follower_id, 0b001000000000)
            for follower_id in vehicle_id_list
        ]

        # vicinity state for all vehicles
        veh_state_neighbour = state_neighbour(vehicle_id_list)

        # follower state for all vehicles, keys are candidate vehicles to control
        veh_state_follower = state_follower(vehicle_id_list)

        # flow_rewards for all edges
        curr_edge_vehicle_count = {
            edge_id: len(traci.edge.getLastStepVehicleIDs(edge_id))
            for edge_id in traci.edge.getIDList()
        }
        flow_rewards = get_flow_reward(prev_edge_vehicle_count, curr_edge_vehicle_count)

        # construct a step
        active_episodes = set()
        batch = {}
        #actions_single = {} # Comment SINGLE
        for vehicle_id, _ in veh_state_follower.items():
            if vehicle_id in ignore_vehicle_ids:
                continue

            current_edge_of_vehicle = traci.vehicle.getRoadID(vehicle_id)
            episode_id = f"{current_edge_of_vehicle}_{vehicle_id}"

            # episodes with MAX_EPISODE_LENGTH steps reached are considered inactive 
            # control for them will resume in the next step
            if episode_id in trajectories and len(trajectories[episode_id]['rewards']) >= MAX_EPISODE_LENGTH:
                continue

            active_episodes.add(episode_id)

            if episode_id not in trajectories:
                trajectories[episode_id] = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "local_rewards": [],
                    "global_rewards": [],
                }

            # step state
            step_gap = veh_state_follower[vehicle_id]["gap"]
            step_follower_speed = veh_state_follower[vehicle_id]["follower_speed"]
            step_leader_speed = veh_state_follower[vehicle_id]["leader_speed"]
            step_count_radius = veh_state_neighbour[vehicle_id]["count_radius"]
            step_mean_speed_radius = veh_state_neighbour[vehicle_id][
                "mean_speed_radius"
            ]
            step_std_speed_radius = veh_state_neighbour[vehicle_id]["std_speed_radius"]
            step_mean_dist_radius = veh_state_neighbour[vehicle_id]["mean_dist_radius"]
            step_std_dist_radius = veh_state_neighbour[vehicle_id]["std_dist_radius"]
            step_green_light = veh_state_follower[vehicle_id]["green_light"]

            # step reward
            step_local_reward = veh_state_follower[vehicle_id]["local_reward"]
            step_global_reward = (
                step_green_light * flow_rewards[current_edge_of_vehicle]
            )
            step_reward = step_local_reward + step_global_reward

            # step action
            step_action = veh_state_follower[vehicle_id]["follower_accel"]

            # trajectories
            trajectories[episode_id]["observations"].append(
                [
                    step_gap,
                    step_follower_speed,
                    step_leader_speed,
                    step_count_radius,
                    step_mean_speed_radius,
                    step_std_speed_radius,
                    step_mean_dist_radius,
                    step_std_dist_radius,
                    step_green_light,
                ]
            )
            trajectories[episode_id]["actions"].append([step_action])
            trajectories[episode_id]["dones"].append(False)
            trajectories[episode_id]["rewards"].append(step_reward)
            trajectories[episode_id]["local_rewards"].append(step_local_reward)
            trajectories[episode_id]["global_rewards"].append(step_global_reward)

            # the sauce
            if len(trajectories[episode_id]['rewards']) > 20:
                batch[vehicle_id] = episode_id
                #pred_act = predict(trajectories[episode_id], 0) # Comment SINGLE
                #actions_single[vehicle_id] = pred_act # Comment SINGLE
                
        #print(f'actions_single {curr_time}', actions_single) # Comment SINGLE

        # batch predict and control
        if len(batch) and DO_CONTROL:
            try: 
                batch_preds = batch_predict([trajectories[episode_id] for _, episode_id in batch.items()], 0)
                actions_batch = {vehicle_id: pa[0] for vehicle_id, pa in zip(batch.keys(), batch_preds)}
                # print(f'actions_batch  {curr_time}', actions_batch)

                # control predicted actions
                for vehicle_id, act in actions_batch.items():
                    traci.vehicle.setAcceleration(vehicle_id, act, 0.04)
            except Exception:
                max_len = max([len(trajectories[episode_id]['rewards']) for _, episode_id in batch.items()])
                print(f'Error while running batch_predict, time: {curr_time}; Max episode len: {max_len}')

        episodes_done = set(trajectories.keys()) - active_episodes

        # delete traj key, keep episode stats
        for ep in episodes_done:
            traj = trajectories[ep]
            if len(traj['observations']) >= 20:
                ep_rew_total = sum([r for r in traj["rewards"]])
                ep_rew_local = sum([r for r in traj["local_rewards"]])
                ep_rew_global = sum([r for r in traj["global_rewards"]])
                ep_len = len([r for r in traj["global_rewards"]])
                episode_stats.append({
                    "episode_id": ep,
                    "ep_rew_total": ep_rew_total,
                    "ep_rew_local": ep_rew_local,
                    "ep_rew_global": ep_rew_global,
                    "ep_len": ep_len
                })
            del trajectories[ep]
 

        prev_edge_vehicle_count = curr_edge_vehicle_count
        curr_time = traci.simulation.getTime()
        previous_vehicle_set = current_vehicle_set

    final_time = traci.simulation.getTime()
    print(f'Finish simulation at time: {final_time} with {len(traci.vehicle.getIDList())} vehicles still in the network')

    # return {k: v for k, v in trajectories.items() if k in episodes_done and len(trajectories[k]['observations']) >= 20}, global_vehicles_exited, global_vehicles_entered
    return episode_stats, global_vehicles_exited, global_vehicles_entered


def stats(traj):
    rewards_1000_ep = []
    local_rewards_1000_ep = []
    global_rewards_1000_ep = []
    ep_len_1000_ep = []
    for i, t in enumerate(trajectories):
        rewards_1000_ep.append(t["ep_rew_total"])
        local_rewards_1000_ep.append(t["ep_rew_local"])
        global_rewards_1000_ep.append(t["ep_rew_global"])
        ep_len_1000_ep.append(t["ep_len"])
    print(f"Num of episodes: {len(traj)}")
    print(f"Mean episode reward = {sum(rewards_1000_ep) / len(rewards_1000_ep)}")
    print(f"Mean episode local reward = {sum(local_rewards_1000_ep) / len(local_rewards_1000_ep)}")
    print(f"Mean episode global reward = {sum(global_rewards_1000_ep) / len(global_rewards_1000_ep)}")
    print(f"Mean episode length = {sum(ep_len_1000_ep) / len(ep_len_1000_ep)}")


# traci.start(["sumo", "-c", "./simulation2/basic_network.sumocfg"])
traci.start(["sumo", "-c", "./big_simulation/conf.sumocfg"])
ignored = warmup()
trajectories, vehicles_exited, vehicles_entered = simulate(ignored)
print(f'Vehicles entered: {len(vehicles_entered)}, Vehicles exited: {len(vehicles_exited)}, Throughtput: {(len(vehicles_exited) - len(vehicles_entered)) / SIMU_MINS} vehs/min')
stats(trajectories)

traci.close()
pickle.dump(
    trajectories,
    open(
        "/project/datasets/test_results_krauss_v5_100m_20181029_d23_1000_1030.pkl",
        "wb",
    ),
)