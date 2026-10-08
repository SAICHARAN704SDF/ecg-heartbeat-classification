from pathlib import Path

DATA_DIR = Path("/Users/charankailasa/Downloads/archive")
TRAIN_FILE = DATA_DIR / "mitbih_train.csv"
TEST_FILE = DATA_DIR / "mitbih_test.csv"

RANDOM_SEED = 42
NUM_CLASSES = 5
SIGNAL_LENGTH = 187

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SIZE = 0.15

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
