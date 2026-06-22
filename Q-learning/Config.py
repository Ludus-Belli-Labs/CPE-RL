# Config

POSTURE_MAP = {"F": 0, "N": 1, "U": 2, "H": 3, "X": 4}

# Rewards & penalties
ATTACK_REWARD             = +10.0   # Reward for attacking a Hostile (H)
ATTACK_PENALTY            = -20.0   # Penalty for attacking anything else
RADAR_REWARD              = +0.5    # Reward for turning radar on
RADAR_OFF_REWARD          = +0.3    # Reward for turning radar off
RADAR_PENALTY             = -0.5    # Penalty for leaving radar on too long without contact
TIME_PENALTY              = -0.001  # Penalty per time unit elapsed
ALLY_DEAD_PENALTY         = -50.0   # Penalty if ally is dead
TARGET_DESTROY_REWARD     = +25.0   # Reward for destroying the target
TARGET_APPROACH_REWARD    = +0.2    # Reward for getting closer to the target
TARGET_RETREAT_PENALTY    = -0.2    # Penalty for getting further from the target
SCENARIO_SUCCESS_REWARD   = +100.0  # Reward for scenario success
SCENARIO_FAIL_PENALTY     = -100.0  # Penalty for scenario failure
CONTACT_CLASSIFIED_REWARD = +0.5    # Reward for classifying a contact from unknown

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