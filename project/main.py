import os
import sys

sys.path.append("/usr/share/sumo/tools")
from model.decision_transformer import predict
import traci

sigma = 0.5

def get_local_reward2(follower_speed, v_safe):
    if follower_speed <= (1 - sigma) * v_safe:
        return -1
    if follower_speed <= v_safe:
        return (1 / sigma / v_safe) * follower_speed - 1 / sigma
    if follower_speed < (1 + sigma) * v_safe:
        return -(1 / sigma / v_safe) * follower_speed + 1 / sigma
    return -1


def warmup():
    previous_vehicle_set = set()
    global_vehicles_exited = set()
    global_vehicles_entered = set()
    while len(global_vehicles_exited) < 1:
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


def simulate(ignore_vehicle_ids: set[str]):
    trajectories = {}
    previous_vehicle_set = set()
    global_vehicles_exited = set()
    global_vehicles_entered = set()

    while len(global_vehicles_exited) < 100:

        traci.simulationStep()
        vehicle_id_list = list(traci.vehicle.getIDList())
        current_vehicle_set = set(vehicle_id_list) - ignore_vehicle_ids
        vehicles_entered = current_vehicle_set - previous_vehicle_set
        vehicles_exited = previous_vehicle_set - current_vehicle_set

        if vehicles_entered:
            global_vehicles_entered.update(vehicles_entered)

        if vehicles_exited:
            global_vehicles_exited.update(vehicles_exited)

        # global_reward = 0 if (len(vehicles_exited) == len(vehicles_entered)) else -1

        for follower_id in vehicle_id_list:
            traci.vehicle.setLaneChangeMode(follower_id, 0b001000000000)

            if follower_id in ignore_vehicle_ids:
                continue

            if follower_id not in trajectories:
                trajectories[follower_id] = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                }

            leader = traci.vehicle.getLeader(follower_id, 1000.0)
            if leader is None:
                continue

            follower_acceleration = traci.vehicle.getAcceleration(follower_id)
            follower_velocity = traci.vehicle.getSpeed(follower_id)
            # velocities.append(follower_velocity)
            min_gap = traci.vehicle.getMinGap(follower_id)
            leader_id, gap = leader
            leader_velocity = traci.vehicle.getSpeed(leader_id)
            green = True
            leader_max_comfortable_decel = traci.vehicle.getDecel(leader_id)
            follower_max_comfortable_decel = traci.vehicle.getDecel(follower_id)

            kraus_follow_speed = traci.vehicle.getFollowSpeed(follower_id, follower_velocity, gap + min_gap, leader_velocity, leader_max_comfortable_decel)

            # print(kraus_follow_speed, follower_velocity)

            local_reward = get_local_reward2(
                follower_velocity,
                kraus_follow_speed
            )
            trajectories[follower_id]["observations"].append(
                [gap, follower_velocity, leader_velocity]
            )
            trajectories[follower_id]["actions"].append([follower_acceleration])
            trajectories[follower_id]["dones"].append(False)
            trajectories[follower_id]["rewards"].append(local_reward)

            # the sauce
            #if len(trajectories[follower_id]['rewards']) > 20:
            #    # print(kraus_follow_speed, follower_velocity)
            #    pred_act = predict(trajectories[follower_id], 0)
            #    #print(pred_act, traci.vehicle.getAcceleration(follower_id ))
            #    traci.vehicle.setAcceleration(follower_id, pred_act, 0.04)
        previous_vehicle_set = current_vehicle_set
    return trajectories, global_vehicles_exited


def stats(traj, completed_traj_ids):
    rewards_1000_ep = []
    for i, (vehID, traj) in enumerate(trajectories.items()):
        if vehID in completed_traj_ids:
            if i % 10 == 0: print(traj['rewards'], traj['rewards'],)
            rewards_1000_ep.append(sum([r for r in traj["rewards"]]))
    print(f"Mean 1000 episode reward = {sum(rewards_1000_ep) / len(rewards_1000_ep)}")


traci.start(["sumo", "-c", "./simulation2/basic_network.sumocfg"])
ignored = warmup()
trajectories, global_vehicles_exited = simulate(ignored)
stats(trajectories, global_vehicles_exited)
traci.close()
