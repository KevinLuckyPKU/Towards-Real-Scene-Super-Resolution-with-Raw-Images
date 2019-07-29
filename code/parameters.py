########################################################################################################################
####            This file will take all parameters we would use for training and testing                            ####
########################################################################################################################

import os
import numpy as np

# model parameter: train or test
# for training: TRAINING is True
# only for testing: TRAINING is False and TESTING is True
# training with testing: TRAINING and TESTING are required to be True
# only test for real image (raw image): REAL is True
TRAINING = True
TESTING = True
REAL = False

# path parameter: path for training and testing data
TRAINING_DATA_PATH = '../Dataset/MOTION/TRAINING'
TESTING_DATA_PATH = '../Dataset/MOTION/TESTING_NOWB'
REAL_DATA_PATH = '../Dataset/test_for_paper/20190230'
SUBFOLDER_TRAININGDATA = 'TrainingSet'
SUBFOLDER_GROUNDTRUTH = 'GT'
SUBFOLDER_ISP = 'ISP'
SUBFOLDER_RAWIMAGE = 'RAWImage'
RESULT_PATH = '../test_result'
LOG_DIR = '../log_dir'
RES_SAVE_FOLDER = None
FILENAME_REPORT = 'record.txt'

# model parameter: other parameter required no matter for training or testing
CROP_SIZE = 256
EPOCH_NUM = 0
BATCH_SIZE = 6
MAX_EPOCH = 40
PRETRAINED = False
TEST_IMAGE_FOLDER = 'test'
STEP_FILE = 'step.npy'
TRAINING_CAPACITY = len(os.listdir(os.path.join(TRAINING_DATA_PATH, SUBFOLDER_TRAININGDATA)))
TRAINING_TRAIN_FILE = os.listdir(os.path.join(TRAINING_DATA_PATH, SUBFOLDER_TRAININGDATA))
BATCH_PER_EPOCH = np.ceil(float(TRAINING_CAPACITY)/BATCH_SIZE)

# parameters for training and testing
LEARNING_RATE = 2e-4
SWITCH_LEARNING_RATE = 1e-5
SWITCH_EPOCH = 20
SAVE_FREQ = 10
TEST_RATIO = 10
GROWTH_RATE = 16
KERNEL_SIZE_DENSE = 3
KERNEL_SIZE_NORMAL = 3
KERNEL_SIZE_POOLING = 2
BOTTLE_OUTPUT = 256
LAYER_PER_BLOCK_DENSE = 8
DECAY_COEF = 0.96
TEST_STEP = 128
SAVE_RAW = False
TESTING_TRAIN_FILE = os.listdir(os.path.join(TESTING_DATA_PATH, SUBFOLDER_TRAININGDATA))
TESTING_CAPACITY = len(TESTING_TRAIN_FILE)


# change settings for training or testing
if REAL or not TRAINING:
    TRAINING_CAPACITY = 0
    MAX_EPOCH = EPOCH_NUM
if REAL:
    PRETRAINED = True
    TESTING_TRAIN_FILE = os.listdir(REAL_DATA_PATH)
    TESTING_CAPACITY = len(TESTING_TRAIN_FILE)


# To make sure folders for containing results exist
def check_and_make_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

        
check_and_make_folder(RESULT_PATH)
check_and_make_folder(LOG_DIR)
