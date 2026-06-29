import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import CMO_SocketClient
import LuaHandler
from RandomScen import random_scen

from Config import (
    POSTURE_MAP,ATTACK_REWARD, ATTACK_PENALTY,
    RADAR_REWARD, RADAR_OFF_REWARD, RADAR_PENALTY,
    TIME_PENALTY, ALLY_DEAD_PENALTY,TARGET_DESTROY_REWARD, 
    TARGET_APPROACH_REWARD, TARGET_RETREAT_PENALTY,
    SCENARIO_SUCCESS_REWARD, SCENARIO_FAIL_PENALTY,
    CONTACT_CLASSIFIED_REWARD, NO_MOVE_PENALTY, NO_MOVE_STEP,RADAR_SPAM_PENALTY, 
    RADAR_SPAM_WINDOW, RADAR_SPAM_MAX_ON,
    TCP_IP, PORT, SIDE, SIM_STEP, RADIUS_KM
)
from Utils import random_point_around, haversine_km

# Environment 

class CPEEnv(gym.Env):
    N_ACTIONS = 10 
    
    def __init__(self):
        super().__init__()

        self.tcp_ip   = TCP_IP
        self.tcp_port = PORT
        self.side     = SIDE
        self.radius_km = RADIUS_KM
        self.sim_step  = SIM_STEP
        self.radar_on = False
        self.radar_steps = 0
        self.is_moving = False
        self.waypoint = None
        self.idle_steps = 0

        self.observation_space = spaces.Box(low=0, high=10, shape=(10,), dtype=np.int64)   
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
        self.prev_ally_dist = None
        self.known_postures = {}
        self.radar_toggle_history = []
        self.attack_slot = -1

    # -- Socket -------------------------------------------------------------

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
        print("[SIM] WARNING: could not confirm sim step completion after 5 attempts — continuing.")

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
    
    # -- What the agent sees -------------------------------------------------
    def _build_obs(self, contacts):
        obs = np.zeros(10, dtype=np.int64)

        # 0-3: contact posture codes (0 if no contact in slot)
        for i, c in enumerate(contacts[:4]):
            obs[i] = c["posture_code"]

        # 4: is_moving (0=not moving, 1=moving)
        obs[4] = int(self.is_moving)

        # 5: radar on/off (0=off, 1=on)
        obs[5] = int(self.radar_on)

        # 6: distance to ally (0=unknown, 1=<10km, 2=10-35km, 3=35-60km, 4=>60km)
        obs[6] = 0
        ally_resp = self._send(LuaHandler.GetAllyUnitCoords(self.unit_name, self.side))
        ally = None
        if isinstance(ally_resp, list) and len(ally_resp) > 0:
            ally = ally_resp[0]
        elif isinstance(ally_resp, dict) and len(ally_resp) > 0:
            first = list(ally_resp.values())[0]
            if isinstance(first, dict) and "latitude" in first:
                ally = first
        if ally and "latitude" in ally and "longitude" in ally:
            d = haversine_km(self.lat, self.lon, float(ally["latitude"]), float(ally["longitude"]))
            if   d < 10:  obs[6] = 1
            elif d < 35:  obs[6] = 2
            elif d < 60:  obs[6] = 3
            else:         obs[6] = 4

        # 7: distance to target (0=unknown, 1=<1000km, 2=1000-1500km, 3=1500-2000km, 4=>2000km)
        obs[7] = 0
        if self.target_guid:
            target_coords = self._get_target_coords()
            if target_coords:
                t_lat, t_lon = target_coords
                d = haversine_km(self.lat, self.lon, t_lat, t_lon)
                if   d < 1000:   obs[7] = 1
                elif d < 1500:  obs[7] = 2
                elif d < 2000:  obs[7] = 3
                else:          obs[7] = 4

        # 8: mission type (0=Unknown, 1=Patrol, 2=Strike)
        mission_map = {"Unknown": 0, "Patrol": 1, "Strike": 2}
        obs[8] = mission_map.get(self.mission_type, 0)

        #9: time remaining (0=unknown, 1=>75%, 2=50-75%, 3=25-50%, 4=<25%)
        obs[9] = 0
        if self.time_limit > 0:
            time_resp = self._send(LuaHandler.GetRemainingTime())
            remaining = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
            ratio = remaining / self.time_limit
            if   ratio > 0.75: obs[9] = 1
            elif ratio > 0.50: obs[9] = 2
            elif ratio > 0.25: obs[9] = 3
            else:              obs[9] = 4

        # 10-13: last attacked contact slot (0-3), or all zeros if no attack yet
        # for i in range(4):
        #     obs[10 + i] = 1 if self.attack_slot == i else 0

        # # 14: can turn radar ON  (1 = radar is currently off, action 7 is valid)
        # obs[14] = int(not self.radar_on)
        # # 15: same as 14 but for radar OFF (1 = radar is currently on,  action 8 is valid)
        # obs[15] = int(self.radar_on)

        return obs

    def _get_coords(self):
        resp = self._send(LuaHandler.GetUnitCoords(self.unit_name, self.side))
        if isinstance(resp, dict) and "latitude" in resp:
            return float(resp["latitude"]), float(resp["longitude"])
        return self.lat, self.lon
    
    def _get_target_coords(self):
        if not self.target_guid:
            print("  [TARGET] No target guid set.")
            return None

        resp = self._send(LuaHandler.GetUnitCoordsByGuid(self.target_guid))
        if isinstance(resp, dict) and "latitude" in resp and "longitude" in resp:
            return float(resp["latitude"]), float(resp["longitude"])

        print(f"  [TARGET] Could not get coords for guid={self.target_guid}.")
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
        self.waypoint = None
        self.is_moving = False
        self.prev_ally_dist = None
        self.known_postures = {}
        self.prev_target_dist = None
        self.idle_steps = 0
        self.radar_toggle_history = []
        self.attack_sloth = -1

        # Ensure connection exists
        if self.client is None:
            self.client = CMO_SocketClient.CMO_SocketClient(self.tcp_ip, self.tcp_port)
            self.client.__enter__()
            if hasattr(self.client, 'socket'):
                self.client.socket.settimeout(30.0)

        # Set time compression
        self._send(LuaHandler.ChangeTimeCompression(4))
        print(f"[RESET] Time compression set to 3")

        # initial score
        resp_score = self._send(LuaHandler.Score(self.side))
        print(f"[RESET] Initial score: {resp_score}")

        # controllable unit 
        resp = self._send(LuaHandler.GetControllableUnit(self.side))
        self.unit_name = resp.get("unitname") if isinstance(resp, dict) else None
        if not self.unit_name:
            raise RuntimeError("No controllable unit found.")
        print(f"[RESET] Unit: {self.unit_name}")

        # time limit 
        time_resp = self._send(LuaHandler.GetRemainingTime())
        self.time_limit = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        time_fmt = time_resp.get('formatted', '?') if isinstance(time_resp, dict) else '?'
        if self.time_limit <= 0:
            raise RuntimeError(f"Invalid time limit: {self.time_limit}")
        print(f"[RESET] Time limit verified: {time_fmt}")

        # Get current coordinates
        self.lat, self.lon = self._get_coords()
        self.contacts = self._get_contacts()

        # mission type 
        mission_resp = self._send(LuaHandler.GetMissionType(self.side))
        fetched_mission = mission_resp.get("[1]", "unknown") if isinstance(mission_resp, dict) else "unknown"
        if fetched_mission == "unknown":
            print(f"[RESET] WARNING: Mission type not yet available, keeping: {self.mission_type}")
        else:
            self.mission_type = fetched_mission
        print(f"[RESET] Mission type verified: {self.mission_type}")

        # target
        target_resp = self._send(LuaHandler.GetTarget(self.side))
        self.target_guid = None
        if isinstance(target_resp, dict) and target_resp:
            first = list(target_resp.values())[0]
            if isinstance(first, str):
                self.target_guid = first
            elif isinstance(first, dict) and first:
                self.target_guid = list(first.values())[0]
        self.prev_target_dist = None
        print(f"[RESET] Target guid verified: {self.target_guid}")

        obs = self._build_obs(self.contacts)
        return obs, {}

    def compute_rewards_and_punishments(self, prev_contacts, attacked_guid, attacked_posture):
        reward = 0.0

        # Radar penalty
        if self.radar_on:
            if len(self.contacts) == 0:
                self.steps += 1
            if self.steps > 10:
                reward += RADAR_PENALTY
                print(f"  [RADAR PENALTY] On for {self.steps} steps without contact.")
        else:
            self.steps = 0

        # Idle / not-moving penalty
        if self.is_moving:
            self.idle_steps = 0
        else:
            self.idle_steps += 1
            if self.idle_steps >= NO_MOVE_STEP:
                reward += NO_MOVE_PENALTY
                print(f"  [IDLE PENALTY] Not moving for {self.idle_steps} steps.")
                self.idle_steps = 0

        # Waypoint reached check
        if self.is_moving and self.waypoint is not None:
            dist_to_wp = haversine_km(self.lat, self.lon, self.waypoint[0], self.waypoint[1])
            if dist_to_wp <= 1.0:
                print(f"  [WAYPOINT] Reached destination ({dist_to_wp:.2f} km). Movement unlocked.")
                self.is_moving = False
                self.waypoint  = None

        # Contact classification reward
        for c in self.contacts:
            guid    = c["guid"]
            posture = c["posture_str"]
            prev_posture = prev_contacts.get(guid) or self.known_postures.get(guid, "U")
            if prev_posture == "U" and posture != "U":
                reward += CONTACT_CLASSIFIED_REWARD
                self.known_postures[guid] = posture
                print(f"  [CLASSIFY] {c['name']} identified as '{posture}' → +{CONTACT_CLASSIFIED_REWARD}")

        # Attack result
        if attacked_guid is not None:
            guids_after = {c["guid"] for c in self.contacts}
            if attacked_guid not in guids_after:
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

        # Check Target distance
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


        # Check ally distance
        if self.mission_type == "Patrol":
            ally_resp = self._send(LuaHandler.GetAllyUnitCoords(self.unit_name, self.side))
            ally = None
            if isinstance(ally_resp, list) and len(ally_resp) > 0:
                ally = ally_resp[0]
            elif isinstance(ally_resp, dict) and len(ally_resp) > 0:
                first = list(ally_resp.values())[0]
                if isinstance(first, dict) and "latitude" in first:
                    ally = first

            if ally and "latitude" in ally and "longitude" in ally:
                dist_to_ally_km = haversine_km(
                    self.lat, self.lon,
                    float(ally["latitude"]), float(ally["longitude"])
                )
                if self.prev_ally_dist is not None:
                    if dist_to_ally_km < self.prev_ally_dist:
                        reward += TARGET_APPROACH_REWARD
                        print(f"  [PATROL] Closer to ally ({dist_to_ally_km:.1f} km, was {self.prev_ally_dist:.1f} km) -> +{TARGET_APPROACH_REWARD}")
                else:
                    print(f"  [PATROL] First ally distance reading: {dist_to_ally_km:.1f} km")

                self.prev_ally_dist = dist_to_ally_km

        return reward

    def step(self, action):
        reward = 0.0
        attacked_guid = None
        attacked_posture = None
        wp_lat, wp_lon = None, None

        MOVE_ACTIONS = {1, 2, 9}  # move_random, move_to_ally, move_to_target
        if self.is_moving and action in MOVE_ACTIONS:
            print(f"  [BLOCKED] Move action {action} ignored — already en-route to {self.waypoint}.")
            reward -= 5.0  
            self.run_simulation()
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
                # self.attack_slot = slot
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
                # self.attack_slot = slot
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
                #   self.attack_slot = slot
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
                #   self.attack_slot = slot
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
                self.radar_toggle_history.append(self.steps)
                self.radar_toggle_history = [
                    s for s in self.radar_toggle_history
                    if self.steps - s <= RADAR_SPAM_WINDOW
                ]
                print(f"  [RADAR ON] Radar activated.")
                reward += RADAR_REWARD
                if len(self.radar_toggle_history) > RADAR_SPAM_MAX_ON:
                    reward += RADAR_SPAM_PENALTY
                    print(f"  [RADAR SPAM PENALTY] {len(self.radar_toggle_history)} toggles in last {RADAR_SPAM_WINDOW} steps.")
            else:
                reward += RADAR_PENALTY
                print(f"  [RADAR ON] Radar already on. Penalty applied.")
        
        elif action == 8: # Radar Off
            if self.radar_on:
                self._send(
                    LuaHandler.EnableShipRadar(self.unit_name, "Passive")
                )
                self.radar_on = False
                self.radar_toggle_history.append(self.steps)
                self.radar_toggle_history = [
                    s for s in self.radar_toggle_history
                    if self.steps - s <= RADAR_SPAM_WINDOW
                ]
                print(f"  [RADAR OFF] Radar deactivated.")
                reward += RADAR_OFF_REWARD
                if len(self.radar_toggle_history) > RADAR_SPAM_MAX_ON:
                    reward += RADAR_SPAM_PENALTY
                    print(f"  [RADAR SPAM PENALTY] {len(self.radar_toggle_history)} toggles in last {RADAR_SPAM_WINDOW} steps.")
            else:
                reward += RADAR_PENALTY
                print(f"  [RADAR OFF] Radar already off. Penalty applied.")

        # elif action == 9 and self.is_moving:   # FullStop while moving
        #     self.is_moving = False
        #     self.waypoint  = None
        #     print("  [SPEED] Set to FullStop.")
        
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

# -- Actions for Strike mission type ---------------------
        elif action == 9 and self.mission_type == "Strike":
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



        self.run_simulation()  
 
        # Update state
        prev_contacts = {c["guid"]: c["posture_str"] for c in self.contacts}
        self.contacts = self._get_contacts()
        self.lat, self.lon = self._get_coords()
        obs = self._build_obs(self.contacts)
        reward += self.compute_rewards_and_punishments(prev_contacts, attacked_guid, attacked_posture)
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
        print("-- Loading new scenario -------------------------------------------------------------------")
        max_reload_attempts = 5
        
        for reload_attempt in range(max_reload_attempts):
            print(f"\n[RESTART] Reload attempt {reload_attempt + 1}/{max_reload_attempts}")
            
            # Load random scenario
            Xml, name = random_scen()
            self.scenario_name = name
            print(f"[RESTART] Scenario: {name}")
            self._send(LuaHandler.RestartScenario(Xml))
            
            time.sleep(3)
            self.reconnect()
            time.sleep(2)

            # Wait for score to reset to 0 (indicates scenario actually loaded fresh)
            print(f"[RESTART] Waiting for score to reset...")
            score_ok = False
            for score_attempt in range(15):
                resp_score = self._send(LuaHandler.Score(self.side))
                score = resp_score.get("score", 0) if isinstance(resp_score, dict) else 0
                if score == 0:
                    print(f"[RESTART] Score verified at 0 after {score_attempt + 1} attempt(s)")
                    score_ok = True
                    break
                print(f"[RESTART] Attempt {score_attempt + 1}/15: Score still {score}, waiting...")
                time.sleep(1)
            
            if not score_ok:
                print(f"[RESTART] Score never reached 0 (still {score}). Possible duplicate TimelineID. Reloading...")
                if reload_attempt < max_reload_attempts - 1:
                    time.sleep(2)
                    continue
                else:
                    print(f"[RESTART] WARNING: Could not get score to 0 after {max_reload_attempts} reloads. Continuing anyway.")
                    break

            # Reset all variables from the fresh scenario (with retry logic)
            max_retries = 10
            reset_ok = False
            for attempt in range(max_retries):
                try:
                    self.reset()
                    print(f"[RESTART] ✓ Scenario ready after {attempt + 1} attempt(s)")
                    reset_ok = True
                    break
                except RuntimeError as e:
                    print(f"[RESTART] Attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            
            if reset_ok:
                return
            
            # If reset failed, try reloading scenario again
            if reload_attempt < max_reload_attempts - 1:
                print(f"[RESTART] Reset failed. Reloading scenario...")
                time.sleep(3)
                continue
            else:
                print(f"[RESTART] WARNING: Could not reset after {max_reload_attempts} reloads. Continuing anyway.")
 
    def check_and_restart(self, attack_posture=None) -> tuple[float, bool, bool]:
        resp_score = self._send(LuaHandler.Score(self.side))
        score = resp_score.get("score", 0) if isinstance(resp_score, dict) else 0
        # print(f"[DEBUG] Score: {score}")
 
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

                    
                    if dist_to_ally_km > 35.0:
                        print(f"[TERMINAL] Patrol FAILED! Merchant reached destination but agent was {dist_to_ally_km:.1f} km away.")
                        self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                        self.restart()
                        return SCENARIO_FAIL_PENALTY, True, False
                    
                    elif dist_to_ally_km <= 35.0:
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
                    penalty = ALLY_DEAD_PENALTY + SCENARIO_FAIL_PENALTY
                    print(f"[TERMINAL] Patrol failed! Ally was hit! Penalty: {penalty}")
                    self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
                    self.restart()
                    return penalty, True, False
                
            
# -- Strike mission checks -----------------------------------------    
        elif self.mission_type == "Strike":
            # Time limit 
            if score == -20:
                elapsed = self.time_limit - remaining
                time_penalty = elapsed * TIME_PENALTY
                full_penalty = SCENARIO_FAIL_PENALTY + time_penalty
                print(f"[TERMINAL] Scenario failed — time limit expired! Penalty: {time_penalty}")
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
            
        elif score == 20 or score == -20 and self.mission_type == "Unknown":
            print(f"[TERMINAL] Scenario ended with Unknown mission")
            self._send('VP_RunForTimeAndHalt ( {Time="00.10.00"} )', fmt="string")
            self.restart()
            return 0.0, True, False

        return 0.0, False, False

        

    def close(self):
        if self.client:
            self.client.__exit__(None, None, None)
            self.client = None
