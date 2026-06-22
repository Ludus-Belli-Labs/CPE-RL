import math
import random
import json
import time
import os
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import CMO_SocketClient
import LuaHandler
from RandomScen import random_scen
from TrainingGraphs import TrainingGraph


# ── Helper ────────────────────────────────────────────────────

def random_point_around(lat, lon, radius_km):
    EARTH_RADIUS = 6371
    distance = radius_km * math.sqrt(random.random())
    bearing  = random.uniform(0, 360)

    lat_rad, lon_rad, bearing_rad = map(math.radians, [lat, lon, bearing])
    ang = distance / EARTH_RADIUS

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(ang)
        + math.cos(lat_rad) * math.sin(ang) * math.cos(bearing_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang) * math.cos(lat_rad),
        math.cos(ang) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# Config -----------------------------------------------------------------


POSTURE_MAP  = {"F": 0, "N": 1, "U": 2, "H": 3, "X": 4}
ATTACK_REWARD   = +10.0   # Reward for attacking a Hostile (H)
ATTACK_PENALTY  = -20.0   # Penalty for attacking anything else
RADAR_REWARD    = +0.5   # Reward for turning radar on
RADAR_OFF_REWARD = +0.3   # Reward for turning radar off 
RADAR_PENALTY   = -0.5   # Penalty for leaving radar on too long without contact
TIME_PENALTY = -0.001  # penalty for time
ALLY_DEAD_PENALTY = -50.0  # Penalty if ally is dead
TARGET_DESTROY_REWARD = +25.0  # Reward for destroying the target
TARGET_APPROACH_REWARD  = +0.2   # Reward for getting closer to the target 
TARGET_RETREAT_PENALTY  = -0.2  # Penalty for getting further from the target
SCENARIO_SUCCESS_REWARD = +100.0  # Reward for scenario success
SCENARIO_FAIL_PENALTY = -100.0  # Penalty for scenario failure
CONTACT_CLASSIFIED_REWARD = +0.5  # Reward for classifying a contact from unknown to something else

# Environment ----------------------------------------------------------------

class CMOEnv(gym.Env):
    N_ACTIONS = 10 
    
    def __init__(self, tcp_ip="127.0.0.1", tcp_port=7778, side="Blue",
                 radius_km=100.0, sim_step="00.05.00", max_steps=30):
        super().__init__()

        self.tcp_ip   = tcp_ip
        self.tcp_port = tcp_port
        self.side     = side
        self.radius_km = radius_km
        self.sim_step  = sim_step
        self.max_steps = max_steps
        self.radar_on = False
        self.radar_steps = 0

        self.observation_space = spaces.Discrete(9)   
        self.action_space      = spaces.Discrete(self.N_ACTIONS)   

        self.client    = None
        self.unit_name = None
        self.lat       = 0.0
        self.lon       = 0.0
        self.steps     = 0
        self.contacts   = []  
        self.time_limit = 0
        self.mission_type = "Unknown"
        self.target_guid  = None       
        self.prev_target_dist = None
        self.known_postures = {}

    # ── Socket helpers ────────────────────────────────────────

    def _send(self, lua, fmt="lua_table", retries=3):
        for attempt in range(retries):
            try:
                self.client.send(lua)
                return self.client.receive(format=fmt)
            except Exception as e:
                print(f"  [SOCKET] Error on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    print(f"  [SOCKET] Reconnecting...")
                    try:
                        self.reconnect()
                    except Exception as re:
                        print(f"  [SOCKET] Reconnect failed: {re}")
                    time.sleep(1)
                else:
                    print(f"  [SOCKET] All retries exhausted, returning safe fallback.")
        return {} if fmt == "lua_table" else ""
    

    def run_simulation(self):
        for attempt in range(5):
            try:
                self.client.send(f'VP_RunForTimeAndHalt({{Time="{self.sim_step}"}})')
                self.client.receive(format="string")
                time.sleep(1)
                return
            except Exception as e:
                print(f"  [SIM] Timeout/error on attempt {attempt+1}/5: {e} — retrying...")
                try:
                    self.reconnect()
                except Exception as re:
                    print(f"  [SIM] Reconnect failed: {re}")
                time.sleep(2)
        print("  [SIM] WARNING: could not confirm sim step completion after 5 attempts — continuing.")

    # ── Getters ───────────────────────────────────────────────

    def _get_contacts(self):
        resp = self._send(LuaHandler.GetContact(self.side))
        contacts = []
        if isinstance(resp, dict) and "result" in resp:
            for line in resp["result"].strip().split("\n"):
                if not line.strip():
                    continue
                name    = line.split("Checking contact:")[1].split("|")[0].strip() if "Checking contact:" in line else None
                posture = line.split("Posture:")[1].split("|")[0].strip()           if "Posture:" in line else "U"
                guid    = line.split("GUID:")[1].split("|")[0].strip()              if "GUID:" in line else None
                if name and guid:
                    contacts.append({
                        "name":         name,
                        "guid":         guid,
                        "posture_code": POSTURE_MAP.get(posture, 2),
                        "posture_str":  posture,
                    })
        return contacts[:4] 
    
    def _build_obs(self, contacts):
        """
        Build observation array from contacts list.
        Slots with no contact default to 0.
        """
        obs = np.zeros(9, dtype=np.int64)
        for i, c in enumerate(contacts):
            obs[i] = c["posture_code"]
        return obs

    def _get_coords(self):
        resp = self._send(LuaHandler.GetUnitCoords(self.unit_name, self.side))
        if isinstance(resp, dict) and "latitude" in resp:
            return float(resp["latitude"]), float(resp["longitude"])
        return self.lat, self.lon
    
    def _get_target_coords(self):
        
        # get target guid
        target_resp = self._send(LuaHandler.GetTarget(self.side))
        guid = None
        if isinstance(target_resp, dict) and target_resp:
            first = list(target_resp.values())[0]
            if isinstance(first, str):
                guid = first
            elif isinstance(first, dict) and first:
                # nested: {'[1]': {'[1]': '<guid>'}}
                guid = list(first.values())[0]
        if not guid:
            print("  [TARGET] Target not found.")
            return None
 
        # find contact name matching the guid
        contact_resp = self._send(LuaHandler.GetContact(self.side))
        target_name = None
        if isinstance(contact_resp, dict) and "result" in contact_resp:
            for line in contact_resp["result"].strip().split("\\"):
                if not line.strip():
                    continue
                name = line.split("Checking contact:")[1].split("|")[0].strip() if "Checking contact:" in line else None
                line_guid = line.split("GUID:")[1].strip() if "GUID:" in line else None
                if name and line_guid and line_guid == guid:
                    target_name = name
                    break
        if not target_name:
            print(f"  [TARGET] No contact found for guid={guid}.")
            return None
 
        # find lat/lon for that name in GetContactCoords
        coords_resp = self._send(LuaHandler.GetContactCoords(self.side))
        if isinstance(coords_resp, dict) and "result" in coords_resp:
            for line in coords_resp["result"].strip().split("\n"):
                if not line.strip():
                    continue
                name = line.split("Checking contact:")[1].split("|")[0].strip() if "Checking contact:" in line else None
                if name == target_name:
                    lat = line.split("Latitude:")[1].split("|")[0].strip()  if "Latitude:"  in line else None
                    lon = line.split("Longitude:")[1].strip()                if "Longitude:" in line else None
                    if lat and lon:
                        return float(lat), float(lon)
 
        print(f"  [TARGET] Coords not found for target '{target_name}'.")
        return None

    # ── Gymnasium API ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
 
        if self.client is None:
            self.client = CMO_SocketClient.CMO_SocketClient(self.tcp_ip, self.tcp_port)
            self.client.__enter__()
            if hasattr(self.client, 'socket'):
                self.client.socket.settimeout(30.0)

        self._send(LuaHandler.ChangeTimeCompression(4))
        print(f"Time compression set to 4")

        resp_score = self._send(LuaHandler.Score(self.side))
        print(f"Initial score: {resp_score}")

        #Get controllable unit
        resp = self._send(LuaHandler.GetControllableUnit(self.side))
        self.unit_name = resp.get("unitname") if isinstance(resp, dict) else None
        if not self.unit_name:
            raise RuntimeError("No controllable unit found.")
        print(f"[ENV] Unit: {self.unit_name}")

        #Get time limit
        time_resp = self._send(LuaHandler.GetRemainingTime())
        self.time_limit = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        print(f"[ENV] Time limit: {time_resp.get('formatted', '?') if isinstance(time_resp, dict) else '?'}")

        self.lat, self.lon = self._get_coords()
        self.contacts = self._get_contacts()

        #Get Mission type
        mission_resp = self._send(LuaHandler.GetMissionType(self.side))
        fetched = mission_resp.get("[1]", "unknown") if isinstance(mission_resp, dict) else "unknown"
        if fetched != "unknown":
            self.mission_type = fetched
        print(f"[ENV] Mission type: {self.mission_type}")

        # Get Target guid 
        target_resp = self._send(LuaHandler.GetTarget(self.side))
        self.target_guid = None
        if isinstance(target_resp, dict) and target_resp:
            first = list(target_resp.values())[0]
            if isinstance(first, str):
                self.target_guid = first
            elif isinstance(first, dict) and first:
                self.target_guid = list(first.values())[0]
        self.prev_target_dist = None
        print(f"[ENV] Target guid: {self.target_guid}")

        obs = self._build_obs(self.contacts)
        return obs, {}

    def step(self, action):
        reward = 0.0
        attacked_guid = None
        attacked_posture = None

        if action == 0:  # nothing
            reward -= 1.0
            print("  [NO ACTION] Doing nothing this step.")
        elif action == 1:  # Move
            wp_lat, wp_lon = random_point_around(self.lat, self.lon, self.radius_km)
            self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
            # self._send(LuaHandler.SetSpeed(self.unit_name, 3))
            print(f"  [MOVE] → ({wp_lat:.3f}, {wp_lon:.3f})")
 
        elif action == 2:  # Move to ally
            resp = self._send(LuaHandler.GetAllyUnitCoords(self.unit_name, self.side))
            ally = None
            if isinstance(resp, list) and len(resp) > 0:
                ally = resp[0]
            elif isinstance(resp, dict) and len(resp) > 0:
                first = list(resp.values())[0]
                if isinstance(first, dict) and "latitude" in first:
                    ally = first
            if ally and "latitude" in ally and "longitude" in ally:
                wp_lat, wp_lon = random_point_around(float(ally["latitude"]), float(ally["longitude"]), radius_km=10.0)
                self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
                # self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                print(f"  [MOVE TO ALLY] {ally.get('name', '?')} → ({wp_lat:.3f}, {wp_lon:.3f})")
            else:
                print("  [MOVE TO ALLY] No allies found.")
 
        elif action == 3:  # Attack contact slot 0
            slot = 0
            if slot < len(self.contacts):
                target = self.contacts[slot]
                attacked_guid = target["guid"]
                attacked_posture = target["posture_str"]
                resp = self._send(LuaHandler.AttackEnemyContact(self.unit_name, self.side, attacked_guid))
                print(f"  [ATTACK slot={slot}] {target['name']} (posture={target['posture_str']}) → {resp}")
            else:
                print(f"  [ATTACK slot={slot}] No contact in this slot.")
                reward += ATTACK_PENALTY * 2
 
        elif action == 4:  # Attack contact slot 1
            slot = 1
            if slot < len(self.contacts):
                target = self.contacts[slot]
                attacked_guid = target["guid"]
                attacked_posture = target["posture_str"]
                resp = self._send(LuaHandler.AttackEnemyContact(self.unit_name, self.side, attacked_guid))
                print(f"  [ATTACK slot={slot}] {target['name']} (posture={target['posture_str']}) → {resp}")
            else:
                print(f"  [ATTACK slot={slot}] No contact in this slot.")
                reward += ATTACK_PENALTY * 2
 
        elif action == 5:  # Attack contact slot 2
            slot = 2
            if slot < len(self.contacts):
                target = self.contacts[slot]
                attacked_guid = target["guid"]
                attacked_posture = target["posture_str"]
                resp = self._send(LuaHandler.AttackEnemyContact(self.unit_name, self.side, attacked_guid))
                print(f"  [ATTACK slot={slot}] {target['name']} (posture={target['posture_str']}) → {resp}")
            else:
                print(f"  [ATTACK slot={slot}] No contact in this slot.")
                reward += ATTACK_PENALTY * 2
 
        elif action == 6:  # Attack contact slot 3
            slot = 3
            if slot < len(self.contacts):
                target = self.contacts[slot]
                attacked_guid = target["guid"]
                attacked_posture = target["posture_str"]
                resp = self._send(LuaHandler.AttackEnemyContact(self.unit_name, self.side, attacked_guid))
                print(f"  [ATTACK slot={slot}] {target['name']} (posture={target['posture_str']}) → {resp}")
            else:
                print(f"  [ATTACK slot={slot}] No contact in this slot.")
                reward += ATTACK_PENALTY * 2

        elif action == 7: # Radar On
            if not self.radar_on:
                self._send(
                    LuaHandler.EnableShipRadar(self.unit_name, "Active")
                )
                self.radar_on = True
                self.steps = 0  
                print(f"  [RADAR ON] Radar activated.")
                reward += RADAR_REWARD
        
        elif action == 8: # Radar Off
            if self.radar_on:
                self._send(
                    LuaHandler.EnableShipRadar(self.unit_name, "Passive")
                )
                self.radar_on = False
                self.steps = 0  # Reset timer
                print(f"  [RADAR OFF] Radar deactivated.")
                reward += RADAR_OFF_REWARD

        # elif action == 9: # set speed to FullStop
        #     self._send(LuaHandler.SetSpeed(self.unit_name, 0))
        #     print(f"  [SPEED] Set to FullStop.")
        
        # elif action == 10: # set speed to Creep
        #     self._send(LuaHandler.SetSpeed(self.unit_name, 1))
        #     print(f"  [SPEED] Set to Creep.")
 
        # elif action == 11: # set speed to Cruise
        #     self._send(LuaHandler.SetSpeed(self.unit_name, 2))
        #     print(f"  [SPEED] Set to Cruise.")

        # elif action == 12: # set speed to Full
        #     self._send(LuaHandler.SetSpeed(self.unit_name, 3))
        #     print(f"  [SPEED] Set to Full.")

        # elif action == 13: # set speed to Flank
        #     self._send(LuaHandler.SetSpeed(self.unit_name, 4))
        #     print(f"  [SPEED] Set to Flank.")

        elif action == 9:  # Move toward target
            target_coords = self._get_target_coords()
            if target_coords:
                t_lat, t_lon = target_coords
                self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, t_lat, t_lon))
                # self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                print(f"  [MOVE TO TARGET] → ({t_lat:.3f}, {t_lon:.3f})")
            else:
                print(f"  [MOVE TO TARGET] Target not found, moving randomly.")
                wp_lat, wp_lon = random_point_around(self.lat, self.lon, self.radius_km)
                self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
                # self._send(LuaHandler.SetSpeed(self.unit_name, 3))
 
 
        if self.radar_on:
                if len(self.contacts) == 0:
                    self.steps += 1
                
                # penalty if radar has been on for more than 10 steps without contact
                if self.steps > 10:
                    reward += RADAR_PENALTY
                    print(f"  [RADAR PENALTY] On for {self.steps} steps without contact.")
        else:
                self.steps = 0

        self.run_simulation()  
 
        # Update state
        prev_contacts = {c["guid"]: c["posture_str"] for c in self.contacts}
        self.contacts = self._get_contacts()
        self.lat, self.lon = self._get_coords()
        obs = self._build_obs(self.contacts)

        # Check for contact classification 
        for c in self.contacts:
            guid    = c["guid"]
            posture = c["posture_str"]
            prev_posture = prev_contacts.get(guid) or self.known_postures.get(guid, "U")
            if prev_posture == "U" and posture != "U":
                reward += CONTACT_CLASSIFIED_REWARD
                self.known_postures[guid] = posture
                print(f"  [CLASSIFY] {c['name']} identified as '{posture}' → +{CONTACT_CLASSIFIED_REWARD}")

        # Check attack result
        if attacked_guid is not None:
            guids_after = {c["guid"] for c in self.contacts}
            if attacked_guid not in guids_after:
                # Contact was destroyed
                if attacked_posture == "H":
                    reward += ATTACK_REWARD
                    print(f"  [REWARD] +{ATTACK_REWARD} — hostile destroyed")
                elif self.mission_type == "Patrol":
                    print(f"  [PATROL] Non-hostile destroyed")
                else:
                    reward += ATTACK_PENALTY
                    print(f"  [PENALTY] {ATTACK_PENALTY} — non-hostile destroyed")
            else:
                print(f"  [ATTACK] Contact still alive")
                

        # Check Target
        if self.target_guid:
            target_coords = self._get_target_coords()
 
            if target_coords is None:
                if self.prev_target_dist is not None:
                    reward += TARGET_DESTROY_REWARD
                    print(f"  [REWARD] +{TARGET_DESTROY_REWARD} — mission target destroyed!")
                    self.target_guid = None
            else:
                t_lat, t_lon = target_coords
                dist_km = haversine_km(self.lat, self.lon, t_lat, t_lon)
                if self.prev_target_dist is not None:
                    if dist_km < self.prev_target_dist:
                        reward += TARGET_APPROACH_REWARD
                        print(f"  [REWARD] +{TARGET_APPROACH_REWARD} — closer to target ({dist_km:.1f} km, was {self.prev_target_dist:.1f} km)")
                    elif dist_km > self.prev_target_dist:
                        reward += TARGET_RETREAT_PENALTY
                        print(f"  [REWARD] {TARGET_RETREAT_PENALTY} — further from target ({dist_km:.1f} km, was {self.prev_target_dist:.1f} km)")
                else:
                    print(f"  [TARGET] First distance reading: {dist_km:.1f} km")
 
                self.prev_target_dist = dist_km


        terminal_reward, terminated, success = self.check_and_restart()
        reward += terminal_reward

        return obs, reward, terminated, False, {"success": success}
    
    def reconnect(self):
        try:
            if self.client:
                self.client.__exit__(None, None, None)
        except Exception:
            pass
        self.client = None
        time.sleep(1)
        self.client = CMO_SocketClient.CMO_SocketClient(self.tcp_ip, self.tcp_port)
        self.client.__enter__()
        if hasattr(self.client, "socket"):
            self.client.socket.settimeout(30)
        print("Reconnected.")
 
    def restart(self):
    
        print("Restarting sim...")
        

        Xml, name = random_scen()
        self.scenario_name = name
        print(f"Loaded: {name} ---------------------------------------------------------------")
        self._send(LuaHandler.RestartScenario(Xml))
        
        time.sleep(1)

        self.reconnect()

        for attempt in range(30):  
            time.sleep(2)
            try:
                time_resp = self._send(LuaHandler.GetRemainingTime())
                remaining = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
                formatted = time_resp.get("formatted", "?") if isinstance(time_resp, dict) else "?"

                unit_resp = self._send(LuaHandler.GetControllableUnit(self.side))
                unit_ready = isinstance(unit_resp, dict) and "unitname" in unit_resp 

                if remaining > 0 :
                    self.time_limit = remaining
                    time.sleep(2)
                    # self._send(f'ScenEdit_SetScore("{self.side}", 0, "Reset")', fmt="string")
                    print(f"CMO ready. New time limit: {formatted}")
                    return
                print(f"Waiting... ({formatted}, unit_ready: {unit_ready}, attempt {attempt + 1}/30)")
            except Exception as e:
                print(f"Exception while waiting for CMO: {e}")
                self.reconnect()
        else:
            print("[RESTART] WARNING: timed out waiting for CMO — proceeding anyway.")

        for attempt in range(10):
            try:
                mission_resp = self._send(LuaHandler.GetMissionType(self.side))
                mission_type = mission_resp.get("[1]", "unknown") if isinstance(mission_resp, dict) else "unknown"
                if mission_type != "unknown":
                    self.mission_type = mission_type
                    print(f"[RESTART] Mission type confirmed: {self.mission_type}")
                    return
                print(f"[RESTART] Waiting for missions... attempt {attempt + 1}/10")
                time.sleep(1)
            except Exception as e:
                print(f"[RESTART] Exception fetching mission type: {e}")
                
        print(f"[RESTART] WARNING: could not confirm mission type — keeping: {self.mission_type}")
 
    def check_and_restart(self) -> tuple[float, bool, bool]:
        resp_score = self._send(LuaHandler.Score(self.side))
        score = resp_score.get("score", 0) if isinstance(resp_score, dict) else 0
        print(f"  [DEBUG] Score: {score}")
 
        time_resp = self._send(LuaHandler.GetRemainingTime())
        remaining = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        formatted = time_resp.get("formatted", "?") if isinstance(time_resp, dict) else "?"

        # mission type check 
        if self.mission_type == "Patrol":
            ally_resp = self._send(LuaHandler.GetAllyAlive(self.unit_name, self.side))
            ally_dead = False
 
            if isinstance(ally_resp, dict):
                if ally_resp.get("alive") is False:
                    ally_dead = True
            elif isinstance(ally_resp, list):
                ally_dead = len(ally_resp) == 0
 
            if ally_dead:
                print(f"[TERMINAL] Failed Patrol mission — ally destroyed! Penalty: {ALLY_DEAD_PENALTY}")

                pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )' 
                self._send(pause, fmt="string")

                self.restart()
                return ALLY_DEAD_PENALTY, True, False
            
        # score check 
        if score == 20:
            elapsed = self.time_limit - remaining
            time_reward = elapsed * TIME_PENALTY
            total_reward = time_reward + SCENARIO_SUCCESS_REWARD
            print(f"[TERMINAL] Scenario complete! {formatted} left → {time_reward:.1f} + {SCENARIO_SUCCESS_REWARD} = {total_reward:.1f} reward")
            pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )' 
            self._send(pause, fmt="string")
            self.restart()
            return total_reward, True, True
        
        elif score == -20:
            if self.mission_type == "Patrol":
                print(f"[TERMINAL] Failed Patrol mission — Attacked a non hostiele unit! Penalty: {SCENARIO_FAIL_PENALTY}")
                
                pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )' 
                self._send(pause, fmt="string")
                self.restart()
                return SCENARIO_FAIL_PENALTY, True, False
            else:
                full_penalty = self.time_limit * TIME_PENALTY
                print(f"[TERMINAL] Time expired! Penalty: {full_penalty}")
                
                pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )' 
                self._send(pause, fmt="string")
                self.restart()
                return full_penalty, True, False
 
        return 0.0, False, False

    def close(self):
        if self.client:
            self.client.__exit__(None, None, None)
            self.client = None


# ── Q-Learning Agent ──────────────────────────────────────────

class QLearningAgent:
    def __init__(self, n_actions=CMOEnv.N_ACTIONS, lr=0.05, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.995):
        self.n_actions     = n_actions
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = {}
        self._last_qtable_size = 0
        self._ep_td_errors = []   
        self._ep_q_values  = []   
 
    def _get_q(self, state):
        key = tuple(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]
 
    def act(self, state):
        key = tuple(int(x) for x in state)
        if key not in self.q_table:
            return random.randrange(self.n_actions)
        q = self._get_q(state)
        if np.all(q == 0):
            return random.randrange(self.n_actions)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(q))
 
    def update(self, state, action, reward, next_state, done):
        q         = self._get_q(state)
        q_next    = self._get_q(next_state)
        target    = reward + (0 if done else self.gamma * np.max(q_next))
        td_error  = target - q[action]
        q[action] += self.lr * td_error
        self._ep_td_errors.append(td_error)
        self._ep_q_values.append(float(np.max(q)))
 
    def decay(self):
        if len(self.q_table) > self._last_qtable_size:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self._last_qtable_size = len(self.q_table)
 
    def save(self, path="model.json"):
        data = {
            "hyperparameters": {
                "n_actions":     self.n_actions,
                "lr":            self.lr,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AGENT] Model saved → {path} ({len(self.q_table)} states, ε={self.epsilon:.4f})")
 
    def save_checkpoint(self, episode: int, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"model_ep{episode}.json")
        self.save(path)
        print(f"[AGENT] Checkpoint saved → {path}")
 
    def load(self, path="model.json"):
        with open(path) as f:
            data = json.load(f)
        if "hyperparameters" in data:
            hp = data["hyperparameters"]
            self.n_actions     = hp.get("n_actions",     self.n_actions)
            self.lr            = hp.get("lr",            self.lr)
            self.gamma         = hp.get("gamma",         self.gamma)
            self.epsilon       = hp.get("epsilon",       self.epsilon)
            self.epsilon_min   = hp.get("epsilon_min",   self.epsilon_min)
            self.epsilon_decay = hp.get("epsilon_decay", self.epsilon_decay)
            self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
            self._last_qtable_size = len(self.q_table)
 
            print(f"[AGENT] Model loaded -----------------------------------------------")
        else:
            self.q_table = {eval(k): np.array(v) for k, v in data.items()}
            self._last_qtable_size = len(self.q_table)
        print(f"[AGENT] Model loaded ← {path} ({len(self.q_table)} states, ε={self.epsilon:.4f})")
 


ACTION_NAMES = {
    0:  "nothing",
    1:  "move_random",
    2:  "move_to_ally",
    3:  "attack_slot_0",
    4:  "attack_slot_1",
    5:  "attack_slot_2",
    6:  "attack_slot_3",
    7:  "radar_on",
    8:  "radar_off",
    # 9:  "speed_fullstop",
    # 10: "speed_creep",
    # 11: "speed_cruise",
    # 12: "speed_full",
    # 13: "speed_flank",
    9: "move_to_target",
}

def _parse_start_episode(load_model: str) -> int:
    import re
    if load_model:
        match = re.search(r"ep(\d+)", os.path.basename(load_model))
        if match:
            return int(match.group(1))
    return 0
 
def train(n_episodes: int, scenario_xml: str = "", load_model: str = None, checkpoint: int = 10):
    env   = CMOEnv()
    agent = QLearningAgent()
    
    n_actions = CMOEnv.N_ACTIONS
    start_episode = 0

    if load_model:                 
        agent.load(load_model)
        start_episode= _parse_start_episode(load_model)
        print(f"[TRAIN] Resuming from: {load_model}")
 
    graph = TrainingGraph(start_episode=start_episode)
    successes = 0
 
    for episode in range(start_episode + 1,start_episode + n_episodes + 1):
        obs, _        = env.reset()
        total_reward  = 0.0
        terminated    = False
        step_count    = 0
        success       = 0
        action_counts = [0] * n_actions
        ep_start      = time.time()
        current_scenario_name = getattr(env, "scenario_name", f"{episode}")
        agent._ep_td_errors = []
        agent._ep_q_values  = []
        
 
        while not terminated:
            action = agent.act(obs)
            action_counts[action] += 1
            next_obs, reward, terminated, _, info = env.step(action)
            step_count   += 1
            agent.update(obs, action, reward, next_obs, terminated)
            obs           = next_obs
            total_reward  += reward
            if info.get("success"):
                success = 1

        td = np.array(agent._ep_td_errors, dtype=np.float32) if agent._ep_td_errors else np.zeros(1)
        qv = np.array(agent._ep_q_values,  dtype=np.float32) if agent._ep_q_values  else np.zeros(1)
        ep_loss   = float(np.mean(td ** 2))
        ep_q_mean = float(np.mean(qv))
        ep_q_max  = float(np.max(qv))
 
        duration_s = time.time() - ep_start
        successes += success
 
        agent.decay()

        if agent.epsilon <= agent.epsilon_min:
            print(f"[END] Episode {episode} — Reached the minium epsilon threshold.")
            agent.save_checkpoint(episode)
            break

        graph.update(
            episode      = episode,
            reward       = total_reward,
            steps        = step_count,
            success      = success,
            epsilon      = agent.epsilon,
            duration_s   = duration_s,
            action_counts= action_counts,
            action_names = ACTION_NAMES,
            scenario = current_scenario_name,
            loss         = ep_loss,
            q_mean       = ep_q_mean,
            q_max        = ep_q_max,
        )

        if episode % checkpoint == 0:
            agent.save_checkpoint(episode)
 
    env.close()
    agent.save()
    graph.save()
    return agent


def run(model_path: str, n_episodes: int = 10):
    """Run a trained model in the sim — no exploration, no learning."""
    env   = CMOEnv()
    agent = QLearningAgent()
    agent.load(model_path)
    agent.epsilon = 0.0  # pure exploitation, no random actions

    for episode in range(1, n_episodes + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        terminated   = False
        step_count   = 0

        while not terminated:
            action = agent.act(obs)         
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            step_count   += 1

        print(f"[RUN] Episode {episode} — reward: {total_reward:.2f}, steps: {step_count}, success: {info.get('success')}")

    env.close()

if __name__ == "__main__":
    train(n_episodes=11, load_model=None, checkpoint=10)

    # for running a trained model without training
    # run(model_path='checkpoints#2/model_ep300.json', n_episodes=1)