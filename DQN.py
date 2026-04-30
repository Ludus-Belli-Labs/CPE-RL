import math
import random
import time
import os
import numpy as np

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces

import CMO_SocketClient
import LuaHandler
from RandomScen import random_scen
from TrainingGraphs import TrainingGraph

import sys
import logging
from datetime import datetime

# ── Helper ────────────────────────────────────────────────────

log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Redirect all print() calls to the logger
class _PrintToLogger:
    def __init__(self, logger): self.logger = logger
    def write(self, msg):
        if msg.strip(): self.logger.info(msg.rstrip())
    def flush(self): pass

sys.stdout = _PrintToLogger(logging.getLogger())


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
CONTACT_CLASSIFIED_REWARD = +5.0  # Reward for classifying a contact from unknown to something else

# Environment ----------------------------------------------------------------

class CMOEnv(gym.Env):
    N_ACTIONS = 15 
    
    def __init__(self, tcp_ip="127.0.0.1", tcp_port=7778, side="Blue",
                 radius_km=100.0, sim_step="00.05.00"):
        super().__init__()

        self.tcp_ip   = tcp_ip
        self.tcp_port = tcp_port
        self.side     = side
        self.radius_km = radius_km
        self.sim_step  = sim_step
        self.radar_on = False
        self.radar_steps = 0
        self.is_moving = False
        self.waypoint = None

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
    
    def get_enemy_radar_range(self) -> float | None:
        
        resp = self._send(LuaHandler.GetEnemyRange())
        if isinstance(resp, dict) and resp:
            val = list(resp.values())[0]
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        return None

    # ── Gymnasium API ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.waypoint  = None
        self.is_moving = False
 
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
        wp_lat, wp_lon = None, None

        MOVE_ACTIONS = {1, 2, 14}  # move_random, move_to_ally, move_to_target
        if self.is_moving and action in MOVE_ACTIONS:
            print(f"  [BLOCKED] Move action {action} ignored — already en-route to {self.waypoint}.")
            reward -= 5.0  
            self.lat, self.lon = self._get_coords()
            # Check if waypoint reached (within 1 km)
            if self.waypoint is not None:
                dist = haversine_km(self.lat, self.lon, self.waypoint[0], self.waypoint[1])
                if dist <= 1.0:
                    print(f"  [WAYPOINT] Reached destination ({dist:.2f} km). Movement unlocked.")
                    self.is_moving = False
                    self.waypoint  = None
            self.contacts = self._get_contacts()
            obs = self._build_obs(self.contacts)
            terminal_reward, terminated, success = self.check_and_restart()
            reward += terminal_reward
            return obs, reward, terminated, False, {"success": success}

        if action == 0:  # nothing
            reward -= 1.0
            print("  [NO ACTION] Doing nothing this step.")
        elif action == 1:  # Move
            wp_lat, wp_lon = random_point_around(self.lat, self.lon, self.radius_km)
            self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
            self._send(LuaHandler.SetSpeed(self.unit_name, 3))
            self.waypoint  = (wp_lat, wp_lon)
            self.is_moving = True
            print(f"  [MOVE] → ({wp_lat:.3f}, {wp_lon:.3f})")
 
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
                reward += ATTACK_PENALTY 
 
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
                reward += ATTACK_PENALTY 
 
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
                reward += ATTACK_PENALTY 
 
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
                reward += ATTACK_PENALTY 

        elif action == 7: # Radar On
            if not self.radar_on:
                self._send(
                    LuaHandler.EnableShipRadar(self.unit_name, "Active")
                )
                self.radar_on = True
                self.steps = 0  
                print(f"  [RADAR ON] Radar activated.")
                reward += RADAR_REWARD
            else:
                reward = RADAR_PENALTY
                print(f"  [RADAR ON] Radar already on. Penalty applied.")
        
        elif action == 8: # Radar Off
            if self.radar_on:
                self._send(
                    LuaHandler.EnableShipRadar(self.unit_name, "Passive")
                )
                self.radar_on = False
                self.steps = 0  # Reset timer
                print(f"  [RADAR OFF] Radar deactivated.")
                reward += RADAR_OFF_REWARD
            else:
                reward = RADAR_PENALTY
                print(f"  [RADAR OFF] Radar already off. Penalty applied.")

        elif action == 9 and self.is_moving:   # FullStop while moving
            self.is_moving = False
            self.waypoint  = None
            print("  [SPEED] Set to FullStop — movement cancelled.")
        
        elif action == 10: # set speed to Creep
            self._send(LuaHandler.SetSpeed(self.unit_name, 1))
            print(f"  [SPEED] Set to Creep.")
 
        elif action == 11: # set speed to Cruise
            self._send(LuaHandler.SetSpeed(self.unit_name, 2))
            print(f"  [SPEED] Set to Cruise.")

        elif action == 12: # set speed to Full
            self._send(LuaHandler.SetSpeed(self.unit_name, 3))
            print(f"  [SPEED] Set to Full.")

        elif action == 13: # set speed to Flank
            self._send(LuaHandler.SetSpeed(self.unit_name, 4))
            print(f"  [SPEED] Set to Flank.")

# -- Actions for Strike mission type ---------------------
        elif action == 14 and self.mission_type == "Strike":
                target_coords = self._get_target_coords()
                if target_coords:
                    t_lat, t_lon = target_coords
                    self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, t_lat, t_lon))
                    self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                    self.waypoint  = (t_lat, t_lon)
                    self.is_moving = True
                    print(f"  [MOVE TO TARGET] → ({t_lat:.3f}, {t_lon:.3f})")
                else:
                    print(f"  [MOVE TO TARGET] Target not found, moving randomly.")

 # -- Actions for Patrol mission type ---------------------
        elif action == 2 and self.mission_type == "Patrol":
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
                    self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                    self.waypoint  = (wp_lat, wp_lon)
                    self.is_moving = True
                    print(f"  [MOVE TO ALLY] {ally.get('name', '?')} → ({wp_lat:.3f}, {wp_lon:.3f})")
                else:
                    print("  [MOVE TO ALLY] No allies found.")


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
        
        # Check if it reached the waypoint
        if self.is_moving and self.waypoint is not None:
            dist_to_wp = haversine_km(self.lat, self.lon, self.waypoint[0], self.waypoint[1])
            if dist_to_wp <= 1.0:
                print(f"  [WAYPOINT] Reached destination ({dist_to_wp:.2f} km). Movement unlocked.")
                self.is_moving = False
                self.waypoint  = None

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
 
    def check_and_restart(self, attack_posture=None) -> tuple[float, bool, bool]:
        resp_score = self._send(LuaHandler.Score(self.side))
        score = resp_score.get("score", 0) if isinstance(resp_score, dict) else 0
        # print(f"  [DEBUG] Score: {score}")
 
        time_resp = self._send(LuaHandler.GetRemainingTime())
        remaining = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        formatted = time_resp.get("formatted", "?") if isinstance(time_resp, dict) else "?"

        
# -- Patrol mission check -----------------------------------------
        if self.mission_type == "Patrol":
            if score == 20:
                # Check distance to ally
                ally_resp = self._send(LuaHandler.GetAllyUnitCoords(self.unit_name, self.side))
                ally = None
                if isinstance(ally_resp, list) and len(ally_resp) > 0:
                    ally = ally_resp[0]
                elif isinstance(ally_resp, dict) and len(ally_resp) > 0:
                    first = list(ally_resp.values())[0]
                    if isinstance(first, dict) and "latitude" in first:
                        ally = first

                if ally and "latitude" in ally and "longitude" in ally:
                    dist_to_ally_km = haversine_km(self.lat, self.lon, float(ally["latitude"]), float(ally["longitude"]))

                    if dist_to_ally_km > 20.0:
                        print(f"[TERMINAL] Patrol FAILED! Merchant reached destination but agent was {dist_to_ally_km:.1f} km away.")
                        self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                        self.restart()
                        return SCENARIO_FAIL_PENALTY, True, False
                    
                    elif dist_to_ally_km <= 20.0:
                        elapsed = self.time_limit - remaining
                        time_reward = elapsed * TIME_PENALTY
                        total_reward = time_reward + SCENARIO_SUCCESS_REWARD
                        print(f"[TERMINAL] Patrol complete! Agent was within range → {total_reward:.1f} reward")
                        self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                        self.restart()
                        return total_reward, True, True
                
            elif score <= -20:
                # Attack a non hostile 
                if attack_posture is not None and attack_posture !="H":
                    print(f"[TERMINAL] Patrol failed! Non-hostile attacked (posture='{attack_posture}')! Penalty: {SCENARIO_FAIL_PENALTY}")
                    self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                    self.restart()
                    return SCENARIO_FAIL_PENALTY, True, False
                else:
                    # Ally dead
                    print(f"[TERMINAL] Patrol failed! Ally was hit! Penalty: {ALLY_DEAD_PENALTY}")
                    self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                    self.restart()
                    return ALLY_DEAD_PENALTY, True, False
                
            
# -- Strike mission checks -----------------------------------------    
        elif self.mission_type == "Strike":
            # Time limit 
            if score == -20:
                full_penalty = self.time_limit * TIME_PENALTY
                print(f"[TERMINAL] Scenario failed — time limit expired! Penalty: {full_penalty}")
                self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                self.restart()
                return full_penalty, True, False
                
            # Detected 
            if self.target_guid:
                target_coords = self._get_target_coords()
                if target_coords:
                    t_lat, t_lon = target_coords
                    radar_range_km = self.get_enemy_radar_range()
                    if radar_range_km is not None:
                        dist_km = haversine_km(self.lat, self.lon, t_lat, t_lon)
                        if dist_km <= radar_range_km:
                            print(f"[TERMINAL] Scenario failed — detected by enemy radar! Distance {dist_km:.1f} km ≤ range {radar_range_km:.1f} km.")
                            self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                            self.restart()
                            return SCENARIO_FAIL_PENALTY, True, False
            # Success
            if score == 20:
                elapsed = self.time_limit - remaining
                time_reward = elapsed * TIME_PENALTY
                total_reward = time_reward + SCENARIO_SUCCESS_REWARD
                print(
                    f"[TERMINAL] Scenario complete! {formatted} left → "
                    f"{time_reward:.1f} + {SCENARIO_SUCCESS_REWARD} = {total_reward:.1f} reward"
                )
                self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                self.restart()
                return total_reward, True, True

        return 0.0, False, False

    def close(self):
        if self.client:
            self.client.__exit__(None, None, None)
            self.client = None


# ── Deep Q-Network (DQN) Agent ────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- Neural Network definition ---

class _QNetwork(nn.Module):
    """Simple MLP: obs_dim → 128 → 128 → n_actions."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent (with experience replay + target network).

    Replaces QLearningAgent while keeping the same external interface:
      act(state) → action
      update(state, action, reward, next_state, done)
      decay()
      save(path) / load(path)
      save_checkpoint(episode)
    """

    def __init__(
        self,
        obs_dim: int       = 9,           # CMOEnv observation size
        n_actions: int     = CMOEnv.N_ACTIONS,
        lr: float          = 0.001,
        gamma: float       = 0.99,
        epsilon: float     = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int    = 64,
        buffer_size: int   = 10_000,
        target_update_freq: int = 30,     # update target net every N learn() calls
    ):
        self.obs_dim           = obs_dim
        self.n_actions         = n_actions
        self.lr                = lr
        self.gamma             = gamma
        self.epsilon           = epsilon
        self.epsilon_min       = epsilon_min
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")

        # Online network (trained every step) and frozen target network
        self.online_net = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        # Counters / diagnostics
        self._learn_steps   = 0
        self._ep_td_errors  = []
        self._ep_q_values   = []

    # ── Internal helpers ──────────────────────────────────────

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        """Convert a numpy obs array to a float32 tensor on device."""
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _learn(self):
        """Sample a mini-batch and perform one gradient-descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough samples yet

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack([self._obs_to_tensor(s) for s in states])
        next_states = torch.stack([self._obs_to_tensor(s) for s in next_states])
        actions     = torch.tensor(actions,  dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards,  dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,    dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        q_vals = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient through target net)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Diagnostics
        td_errors = (targets - q_vals).detach().cpu().numpy()
        self._ep_td_errors.extend(td_errors.tolist())
        self._ep_q_values.extend(q_vals.detach().cpu().numpy().tolist())

        # Periodically sync target network
        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            print(f"  [DQN] Target network synced at learn-step {self._learn_steps}")

    # ── Public interface (mirrors QLearningAgent) ─────────────

    def act(self, state) -> int:
        """epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online_net(self._obs_to_tensor(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        """Store transition in replay buffer then learn."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        self._learn()

    def decay(self):
        """Decay epsilon after each episode (unconditional, mirrors old behaviour)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="model.pt"):
        """Save the full agent state (weights + hyperparameters) to a .pt file."""
        torch.save({
            "hyperparameters": {
                "obs_dim":            self.obs_dim,
                "n_actions":          self.n_actions,
                "lr":                 self.lr,
                "gamma":              self.gamma,
                "epsilon":            self.epsilon,
                "epsilon_min":        self.epsilon_min,
                "epsilon_decay":      self.epsilon_decay,
                "batch_size":         self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        print(f"[DQN] Model saved → {path} (ε={self.epsilon:.4f}, "
              f"buffer={len(self.replay_buffer)})")

    def save_checkpoint(self, episode: int, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"model_ep{episode}.pt")
        self.save(path)
        print(f"[DQN] Checkpoint saved → {path}")

    def load(self, path="model.pt"):
        """Load a previously saved .pt checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if "hyperparameters" in checkpoint:
            hp = checkpoint["hyperparameters"]
            self.obs_dim            = hp.get("obs_dim",            self.obs_dim)
            self.n_actions          = hp.get("n_actions",          self.n_actions)
            self.lr                 = hp.get("lr",                 self.lr)
            self.gamma              = hp.get("gamma",              self.gamma)
            self.epsilon            = hp.get("epsilon",            self.epsilon)
            self.epsilon_min        = hp.get("epsilon_min",        self.epsilon_min)
            self.epsilon_decay      = hp.get("epsilon_decay",      self.epsilon_decay)
            self.batch_size         = hp.get("batch_size",         self.batch_size)
            self.target_update_freq = hp.get("target_update_freq", self.target_update_freq)

        # Rebuild networks with (possibly updated) dims
        self.online_net = _QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = _QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"[DQN] Model loaded ← {path} (ε={self.epsilon:.4f})")
 


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
    9:  "speed_fullstop",
    10: "speed_creep",
    11: "speed_cruise",
    12: "speed_full",
    13: "speed_flank",
    14: "move_to_target",
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
    agent = DQNAgent()
    
    n_actions = CMOEnv.N_ACTIONS
    start_episode = 0

    if load_model:                 
        agent.load(load_model)
        start_episode= _parse_start_episode(load_model)
        print(f"[TRAIN] Resuming from: {load_model}")
 
    graph = TrainingGraph(start_episode=start_episode)
    successes = 0
    epsilon_restarts     = 0
    MAX_EPSILON_RESTARTS = 3
    EPSILON_RESET_VALUE  = 0.3
 
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
            if epsilon_restarts >= MAX_EPSILON_RESTARTS:
                print(f"[END] Episode {episode} — Reached epsilon minimum after {epsilon_restarts} restarts. Training complete.")
                agent.save_checkpoint(episode)
                break
            epsilon_restarts += 1
            agent.epsilon = EPSILON_RESET_VALUE
            print(f"[EPSILON RESTART #{epsilon_restarts}/{MAX_EPSILON_RESTARTS}] Episode {episode} — "
                  f"resetting epsilon to {EPSILON_RESET_VALUE} and continuing training.")

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
    agent.save("model.pt")
    graph.save()
    return agent


def run(model_path: str, n_episodes: int = 10):
    """Run a trained model in the sim — no exploration, no learning."""
    env   = CMOEnv()
    agent = DQNAgent()
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
    train(n_episodes=10000, load_model=None, checkpoint=10)

    # for running a trained model without training
    # run(model_path='checkpoints/model_ep400.pt', n_episodes=1)