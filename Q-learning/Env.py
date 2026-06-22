import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import CMO_SocketClient
import LuaHandler
from RandomScen import random_scen

from RL.Config import (
    POSTURE_MAP,
    ATTACK_REWARD, ATTACK_PENALTY,
    RADAR_REWARD, RADAR_OFF_REWARD, RADAR_PENALTY,
    TIME_PENALTY,
    ALLY_DEAD_PENALTY,
    TARGET_DESTROY_REWARD, TARGET_APPROACH_REWARD, TARGET_RETREAT_PENALTY,
    SCENARIO_SUCCESS_REWARD, SCENARIO_FAIL_PENALTY,
    CONTACT_CLASSIFIED_REWARD,
)
from RL.Utils import random_point_around, haversine_km

# Environment 

class CMOEnv(gym.Env):
    N_ACTIONS = 15

    def __init__(self, tcp_ip="127.0.0.1", tcp_port=7777, side="Blue",
                 radius_km=100.0, sim_step="00.05.00", max_steps=30):
        super().__init__()

        self.tcp_ip    = tcp_ip
        self.tcp_port  = tcp_port
        self.side      = side
        self.radius_km = radius_km
        self.sim_step  = sim_step
        self.max_steps = max_steps
        self.radar_on  = False
        self.radar_steps = 0

        self.observation_space = spaces.Discrete(9)
        self.action_space      = spaces.Discrete(self.N_ACTIONS)

        self.client           = None
        self.unit_name        = None
        self.lat              = 0.0
        self.lon              = 0.0
        self.steps            = 0
        self.contacts         = []
        self.time_limit       = 0
        self.mission_type     = "Unknown"
        self.target_guid      = None
        self.prev_target_dist = None
        self.known_postures   = {}

    # Socket helpers 
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
        """
        Send VP_RunForTimeAndHalt once. If the socket times out (CMO is still
        simulating), do NOT resend — just wait and poll until the remaining
        time has actually decreased, confirming CMO finished the step.
        """
        try:
            self.client.send(f'VP_RunForTimeAndHalt({{Time="{self.sim_step}"}})')
            self.client.receive(format="string")
            time.sleep(1)
        except Exception as e:
            print(f"  [SIM] Socket timed out waiting for CMO ({e}) — polling for completion...")
            for attempt in range(60):  # max 60s wait
                time.sleep(5)
                try:
                    resp = self._send(LuaHandler.GetRemainingTime(), retries=1)
                    remaining = resp.get("remaining", -1) if isinstance(resp, dict) else -1
                    if remaining >= 0:
                        print(f"  [SIM] CMO step complete after {attempt + 1}s wait.")
                        return
                except Exception:
                    pass
            print(f"  [SIM] WARNING: could not confirm CMO step completion — continuing anyway.")

    # Getters 

    def _get_contacts(self):
        resp = self._send(LuaHandler.GetContact(self.side))
        contacts = []
        if isinstance(resp, dict) and "result" in resp:
            for line in resp["result"].strip().split("\n"):
                if not line.strip():
                    continue
                name    = line.split("Checking contact:")[1].split("|")[0].strip() if "Checking contact:" in line else None
                posture = line.split("Posture:")[1].split("|")[0].strip()           if "Posture:"          in line else "U"
                guid    = line.split("GUID:")[1].split("|")[0].strip()              if "GUID:"             in line else None
                if name and guid:
                    contacts.append({
                        "name":         name,
                        "guid":         guid,
                        "posture_code": POSTURE_MAP.get(posture, 2),
                        "posture_str":  posture,
                    })
        return contacts[:4]

    def _build_obs(self, contacts):
        """Build observation array from contacts list. Empty slots default to 0."""
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
        # Get target GUID
        target_resp = self._send(LuaHandler.GetTarget(self.side))
        guid = None
        if isinstance(target_resp, dict) and target_resp:
            first = list(target_resp.values())[0]
            if isinstance(first, str):
                guid = first
            elif isinstance(first, dict) and first:
                guid = list(first.values())[0]
        if not guid:
            print("  [TARGET] Target not found.")
            return None

        # Find contact name matching the GUID
        contact_resp = self._send(LuaHandler.GetContact(self.side))
        target_name = None
        if isinstance(contact_resp, dict) and "result" in contact_resp:
            for line in contact_resp["result"].strip().split("\\"):
                if not line.strip():
                    continue
                name      = line.split("Checking contact:")[1].split("|")[0].strip() if "Checking contact:" in line else None
                line_guid = line.split("GUID:")[1].strip()                            if "GUID:"             in line else None
                if name and line_guid and line_guid == guid:
                    target_name = name
                    break
        if not target_name:
            print(f"  [TARGET] No contact found for guid={guid}.")
            return None

        # Find lat/lon for that contact name
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

    # Gymnasium API 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        if self.client is None:
            self.client = CMO_SocketClient.CMO_SocketClient(self.tcp_ip, self.tcp_port)
            self.client.__enter__()

        self._send(LuaHandler.ChangeTimeCompression(4))
        print(f"Time compression set to 4")

        resp_score = self._send(LuaHandler.Score(self.side))
        print(f"Initial score: {resp_score}")

        # Get controllable unit
        resp = self._send(LuaHandler.GetControllableUnit(self.side))
        self.unit_name = resp.get("unitname") if isinstance(resp, dict) else None
        if not self.unit_name:
            raise RuntimeError("No controllable unit found.")
        print(f"[ENV] Unit: {self.unit_name}")

        # Get time limit
        time_resp = self._send(LuaHandler.GetRemainingTime())
        self.time_limit = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        print(f"[ENV] Time limit: {time_resp.get('formatted', '?') if isinstance(time_resp, dict) else '?'}")

        self.lat, self.lon = self._get_coords()
        self.contacts = self._get_contacts()

        # Get mission type
        mission_resp = self._send(LuaHandler.GetMissionType(self.side))
        fetched = mission_resp.get("[1]", "unknown") if isinstance(mission_resp, dict) else "unknown"
        if fetched != "unknown":
            self.mission_type = fetched
        print(f"[ENV] Mission type: {self.mission_type}")

        # Get target GUID
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
        attacked_guid    = None
        attacked_posture = None

        # Dispatch action 

        if action == 0:  # Nothing
            print("  [NO ACTION] Doing nothing this step.")

        elif action == 1:  # Move randomly
            wp_lat, wp_lon = random_point_around(self.lat, self.lon, self.radius_km)
            self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
            self._send(LuaHandler.SetSpeed(self.unit_name, 3))
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
                self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                print(f"  [MOVE TO ALLY] {ally.get('name', '?')} → ({wp_lat:.3f}, {wp_lon:.3f})")
            else:
                print("  [MOVE TO ALLY] No allies found.")

        elif action in (3, 4, 5, 6):  # Attack contact slot 0–3
            slot = action - 3
            if slot < len(self.contacts):
                target = self.contacts[slot]
                attacked_guid    = target["guid"]
                attacked_posture = target["posture_str"]
                resp = self._send(LuaHandler.AttackEnemyContact(self.unit_name, self.side, attacked_guid))
                print(f"  [ATTACK slot={slot}] {target['name']} (posture={target['posture_str']}) → {resp}")
            else:
                print(f"  [ATTACK slot={slot}] No contact in this slot.")
                reward += ATTACK_PENALTY

        elif action == 7:  # Radar on
            if not self.radar_on:
                self._send(LuaHandler.EnableShipRadar(self.unit_name, "Active"))
                self.radar_on  = True
                self.steps     = 0
                print(f"  [RADAR ON] Radar activated.")
                reward += RADAR_REWARD

        elif action == 8:  # Radar off
            if self.radar_on:
                self._send(LuaHandler.EnableShipRadar(self.unit_name, "Passive"))
                self.radar_on  = False
                self.steps     = 0
                print(f"  [RADAR OFF] Radar deactivated.")
                reward += RADAR_OFF_REWARD

        elif action == 9:   # Speed: FullStop
            self._send(LuaHandler.SetSpeed(self.unit_name, 0))
            print(f"  [SPEED] Set to FullStop.")

        elif action == 10:  # Speed: Creep
            self._send(LuaHandler.SetSpeed(self.unit_name, 1))
            print(f"  [SPEED] Set to Creep.")

        elif action == 11:  # Speed: Cruise
            self._send(LuaHandler.SetSpeed(self.unit_name, 2))
            print(f"  [SPEED] Set to Cruise.")

        elif action == 12:  # Speed: Full
            self._send(LuaHandler.SetSpeed(self.unit_name, 3))
            print(f"  [SPEED] Set to Full.")

        elif action == 13:  # Speed: Flank
            self._send(LuaHandler.SetSpeed(self.unit_name, 4))
            print(f"  [SPEED] Set to Flank.")

        elif action == 14:  # Move to target
            target_coords = self._get_target_coords()
            if target_coords:
                t_lat, t_lon = target_coords
                self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, t_lat, t_lon))
                self._send(LuaHandler.SetSpeed(self.unit_name, 3))
                print(f"  [MOVE TO TARGET] → ({t_lat:.3f}, {t_lon:.3f})")
            else:
                print(f"  [MOVE TO TARGET] Target not found, moving randomly.")
                wp_lat, wp_lon = random_point_around(self.lat, self.lon, self.radius_km)
                self._send(LuaHandler.SetUnitWaypoint(self.unit_name, self.side, wp_lat, wp_lon))
                self._send(LuaHandler.SetSpeed(self.unit_name, 3))

        # Radar time penalty
        if self.radar_on:
            if len(self.contacts) == 0:
                self.steps += 1
            # Penalty if radar has been on for more than 10 steps without contact
            if self.steps > 10:
                reward += RADAR_PENALTY
                print(f"  [RADAR PENALTY] On for {self.steps} steps without contact.")
        else:
            self.steps = 0

        self.run_simulation()

        # Update state 
        prev_contacts  = {c["guid"]: c["posture_str"] for c in self.contacts}
        self.contacts  = self._get_contacts()
        self.lat, self.lon = self._get_coords()
        obs = self._build_obs(self.contacts)

        # Contact classification reward
        for c in self.contacts:
            guid         = c["guid"]
            posture      = c["posture_str"]
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

        # Target distance reward/penalty
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

    # Connection management 

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
        print("Reconnected.")

    def restart(self):
        print("Restarting sim...")

        xml, name = random_scen()
        self.scenario_name = name
        self._send(LuaHandler.RestartScenario(xml))
        print(f"Loaded: {name} ---------------------------------------------------------------")
        time.sleep(1)

        self.reconnect()

        for attempt in range(30):
            time.sleep(2)
            try:
                time_resp  = self._send(LuaHandler.GetRemainingTime())
                remaining  = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
                formatted  = time_resp.get("formatted", "?") if isinstance(time_resp, dict) else "?"
                unit_resp  = self._send(LuaHandler.GetControllableUnit(self.side))
                unit_ready = isinstance(unit_resp, dict) and "unitname" in unit_resp

                if remaining > 0:
                    self.time_limit = remaining
                    time.sleep(2)
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
        score      = resp_score.get("score", 0) if isinstance(resp_score, dict) else 0
        print(f"  [DEBUG] Score: {score}")

        time_resp = self._send(LuaHandler.GetRemainingTime())
        remaining = time_resp.get("remaining", 0) if isinstance(time_resp, dict) else 0
        formatted = time_resp.get("formatted", "?") if isinstance(time_resp, dict) else "?"

        # Patrol-specific: ally alive check
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
                self.client.send(f"{pause}")
                self.restart()
                return ALLY_DEAD_PENALTY, True, False

        # Score-based terminal conditions
        if score == 20:
            elapsed      = self.time_limit - remaining
            time_reward  = elapsed * TIME_PENALTY
            total_reward = time_reward + SCENARIO_SUCCESS_REWARD
            print(f"[TERMINAL] Scenario complete! {formatted} left → {time_reward:.1f} + {SCENARIO_SUCCESS_REWARD} = {total_reward:.1f} reward")
            pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )'
            self.client.send(f"{pause}")
            self.restart()
            return total_reward, True, True

        elif score == -20:
            if self.mission_type == "Patrol":
                print(f"[TERMINAL] Failed Patrol mission — Attacked a non-hostile unit! Penalty: {SCENARIO_FAIL_PENALTY}")
                pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )'
                self.client.send(f"{pause}")
                self.restart()
                return SCENARIO_FAIL_PENALTY, True, False
            else:
                full_penalty = self.time_limit * TIME_PENALTY
                print(f"[TERMINAL] Time expired! Penalty: {full_penalty}")
                pause = 'VP_RunForTimeAndHalt ( {Time="00.10.00"} )'
                self.client.send(f"{pause}")
                self.restart()
                return full_penalty, True, False

        return 0.0, False, False

    def close(self):
        if self.client:
            self.client.__exit__(None, None, None)
            self.client = None