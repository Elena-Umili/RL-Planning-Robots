
##################### ACROBOT
STATE_SIZE = 6
INPUT_SIZE = STATE_SIZE
ACTION_SIZE = 3
CODE_SIZE = 24
DISCRETE_CODES = False
DISCRETE_LOSS = False
REW_THRE = -120
MINIMUM_REWARD = -500
WINDOW = 20
MODELS_DIR = "models/acrobot/"

Q_HIDDEN_NODES = 96
BATCH_SIZE = 64
MARGIN = 0.1
STANDARD_DEVIATION = 0.2
PREDICT_CERTAINTY = True
