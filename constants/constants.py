import numpy as np

# path to preprocessed images
#TRAIN_PATH = '/mnt/disks/large/data/train_split/train'
TRAIN_PATH = '/mnt/disks/large/debug_data/train'

#DEV_PATH = '/mnt/disks/large/data/train_split/val'
DEV_PATH = '/mnt/disks/large/debug_data/val'

#TEST_PATH = '/mnt/disks/large/data/train_split/test'
TEST_PATH = '/mnt/disks/large/debug_data/test'


# path to preprocessed csv files, containing preprocessed image ids and attribute ids
#TRAIN_CSV = 'data/train_split_train.csv'
#DEV_CSV = 'data/train_split_val.csv'
#TEST_CSV = 'data/train_split_test.csv'
TRAIN_CSV = '/mnt/disks/large/debug_data/train_split_train.csv'
DEV_CSV = '/mnt/disks/large/debug_data/train_split_val.csv'
TEST_CSV = '/mnt/disks/large/debug_data/train_split_test.csv'

# random seed
SEED = 231

# number of classes
NUM_CLASSES = 1103

# per-channel mean & std of training set
MEAN = [144.46578057, 156.0374637 , 165.11049366]
STD = [64.50104194, 63.66100116, 63.80019134]

# culture labels
CULTURE_LABELS = np.arange(0,398)

# number of random crops
NUM_CROPS = 5
