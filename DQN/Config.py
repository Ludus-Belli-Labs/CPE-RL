# CPE configurations
PORT = 7777
SIDE = 'Blue'
TCP_IP = '127.0.0.1'
SIM_STEP = "00.02.00"
RADIUS_KM = 100

# Reward structure for DQN agent
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
NO_MOVE_PENALTY = -5.0  # Penalty for taking no action (to encourage exploration)
NO_MOVE_STEP = 5  # Number of consecutive no-move steps before applying penalty
RADAR_SPAM_PENALTY = -5.0  # Penalty for toggling radar on/off too frequently
RADAR_SPAM_WINDOW = 10  # Number of toggles within 10 steps to be considered spamming
RADAR_SPAM_MAX_ON = 5  # Max number of times radar can be turned on within the threshold before penalty applies

# DQN hyperparameters
LEARNINGRATE = 0.001
EPSILON_MIN = 0.2
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
UPDATE_FREQUENCY = 3  # update target net every N episodes

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