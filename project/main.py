import os
import sys

sys.path.append("/usr/share/sumo/tools")
import traci

traci.start(["sumo", "-c", "./simulation/hello.sumocfg"])

step = 0
while step < 1000:
    traci.simulationStep()
    vehicle_id_list = list(traci.vehicle.getIDList())
    if vehicle_id_list:
        vehicle_id = vehicle_id_list[0]
        print(traci.vehicle.getPosition(vehicle_id))
    step += 1

traci.close()