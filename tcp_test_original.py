import CMO_SocketClient
import time
import json
import LuaHandler
import math
import random
from RandomScen import random_scen

def random_point_around(lat, lon, radius_km):
    EARTH_RADIUS = 6371

    # Random distance with uniform area distribution
    distance = radius_km * math.sqrt(random.random())
    bearing = random.uniform(0, 360)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)

    angular_distance = distance / EARTH_RADIUS

    new_lat = math.asin(
        math.sin(lat_rad)*math.cos(angular_distance) +
        math.cos(lat_rad)*math.sin(angular_distance)*math.cos(bearing_rad)
    )

    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad)*math.sin(angular_distance)*math.cos(lat_rad),
        math.cos(angular_distance)-math.sin(lat_rad)*math.sin(new_lat)
    )

    return math.degrees(new_lat), math.degrees(new_lon)



if __name__ == "__main__":
    TCP_IP = '127.0.0.1'
    TCP_PORT = 7777 

    runSimulation = 'VP_RunForTimeAndHalt ( {Time="00.05.00"} )'
    side = 'Blue' 
    
    try:
        with CMO_SocketClient.CMO_SocketClient(TCP_IP, TCP_PORT) as client:
            while True:
                start_time = time.perf_counter()

                client.send(f"{runSimulation}")
                # print(f" >>> Get Scenario Status")
                # client.send(LuaHandler.GetScenarioStatus()) 
                # print(client.receive(format='lua_table'))
 
                # print(f"\n >>> Get controllable unit ")
                # client.send(LuaHandler.GetControllableUnit(side))
                # unit_response = client.receive(format='lua_table')

                # if isinstance(unit_response, dict) and 'unitname' in unit_response:
                #     unitName = unit_response['unitname']  
                #     print(f"Controllable unit: {unitName}")
                # else:
                #     print("No controllable unit found.")
                #     continue

                print("\n>>> Sending 'Run Simulation' command...")
                
                # client.send(f"{simulationSpeed}")
                
                json_response = client.receive(format='string')
                print("<<< Received JSON data:")
                if json_response:
                    print(json.dumps(json_response, indent=2))
                


                # print ("\n>>> Sending 'Get Unit coords' command...")
                # client.send(LuaHandler.GetUnitCoords(unitName, side))
                # unit_coords = client.receive(format='lua_table')
                # # print (f"\n<<< Received unit coords: {unit_coords}")
                # ship_lat = unit_coords['latitude']
                # ship_lon = unit_coords['longitude']

                # print("\n>>> Sending 'Set Unit Waypoint' command...")
                # wp_lat, wp_lon = random_point_around(ship_lat, ship_lon, radius_km=100)
                # client.send(LuaHandler.SetUnitWaypoint(unitName, side, wp_lat, wp_lon))
                # wp_response = client.receive(format='lua_table')

                # throttle = 3 # 0 = FullStop, 1 = Creep, 2 = Cruise, 3 = Full, 4 = Flank, 5  = None

                # client.send(LuaHandler.ChangeTimeCompression(3))

                client.send(LuaHandler.ChangeTimeCompression(3))

                # print(f"\n >>> Set speed")
                # client.send(LuaHandler.SetSpeed(unitName, throttle))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Enable ship radar")
                # client.send(LuaHandler.EnableShipRadar(unitName, "Active"))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Contact info")
                # client.send(LuaHandler.GetContact(side))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Contact coords")
                # client.send(LuaHandler.GetContactCoords(side))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Attack contact")
                # client.send(LuaHandler.AttackEnemyContact(unitName, side))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Get Damage")
                # client.send(LuaHandler.GetDamage(unitName))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Set time speed")
                # client.send(LuaHandler.ChangeTimeCompression(5))
                # print(client.receive(format='lua_table'))

                # print(f" >>> Get Scenario Info")
                # client.send(LuaHandler.GetRemainingTime())
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Get Ally Units")
                # client.send(LuaHandler.GetAllyUnitCoords(unitName, side))
                # print(client.receive(format='lua_table'))


                # print("\n>>> Get Ally Units")
                # client.send(LuaHandler.GetAllyUnitCoords(unitName, side))
                # ally_units = client.receive(format='lua_table')
                # print(ally_units)

                # # Get first ally
                # ally = list(ally_units.values())[0]

                # ally_lat = ally['latitude']
                # ally_lon = ally['longitude']

                # # waypoint around ally
                # wp_lat, wp_lon = random_point_around(ally_lat, ally_lon, radius_km=10)

                # print("\n>>> Sending 'Set Unit Waypoint' command...") 
                # client.send(LuaHandler.SetUnitWaypoint(unitName, side, wp_lat, wp_lon))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Get Mission")
                # client.send(LuaHandler.GetMissionType(side))
                # print(client.receive(format='lua_table'))

                # print(f"\n >>> Get Target")
                # client.send(LuaHandler.GetTarget(side))
                # print(client.receive(format='lua_table'))

                print(f"\n >>> Get Score")
                client.send(LuaHandler.Score(side))
                score = client.receive(format='lua_table')
                print(f"\n<<< Received score: {score}")
                # if score.get('score') == 20 or score.get('score') == -20:
                #     xml,name = random_scen()
                #     print(f"\n >>> Restart Scenario: {name}")
                #     client.send(LuaHandler.RestartScenario(xml))
                #     print(client.receive(format='lua_table'))
                # else:
                #     print(f"\n Score is not 20")


                # print(f"\n >>> Get Target")
                # client.send(LuaHandler.GetTarget(side))
                # print(client.receive(format='lua_table'))

            

                end_time = time.perf_counter()
                latency = end_time - start_time
                print(f"\nTotal cycle time: {latency:.4f} seconds")
                print("-" * 40)
                time.sleep(1)
    except ConnectionRefusedError:
        print(f"\n[ERROR] Connection failed. Is a server running on {TCP_IP}:{TCP_PORT}?")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")