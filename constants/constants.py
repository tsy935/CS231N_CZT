import numpy as np

# path to preprocessed images
TRAIN_PATH = '/mnt/disks/large/data/train_split/train'
#TRAIN_PATH = '/mnt/disks/large/debug_data/train'

DEV_PATH = '/mnt/disks/large/data/train_split/val'
#DEV_PATH = '/mnt/disks/large/debug_data/val'

TEST_PATH = '/mnt/disks/large/data/train_split/test'
#TEST_PATH = '/mnt/disks/large/debug_data/test'


# path to preprocessed csv files, containing preprocessed image ids and attribute ids
TRAIN_CSV = 'data/train_split_train.csv'
DEV_CSV = 'data/train_split_val.csv'
TEST_CSV = 'data/train_split_test.csv'
#TRAIN_CSV = '/mnt/disks/large/debug_data/train_split_train.csv'
#DEV_CSV = '/mnt/disks/large/debug_data/train_split_val.csv'
#TEST_CSV = '/mnt/disks/large/debug_data/train_split_test.csv'

# args file name
ARGS_FILE_NAME = 'args.json'

# random seed
SEED = 231

# number of classes
NUM_CLASSES = 1103

# per-channel mean & std of training set
MEAN = [0.56584825, 0.61147285, 0.6472829]
STD = [0.25330281, 0.25008629, 0.25073694]

# culture labels
CULTURE_LABELS = list(np.arange(0,398))

# number of random crops
NUM_CROPS = 5
#NUM_CROPS = 0