# centralized configuration file.
# It can be updated using argparse as well.


class AE_setting:
    EMBEDDING = 128
    NUM_CHANNELS = 2
    STRIDE = 2  # This is fixed
    KERNEL_SIZE = 4  # This is fixed
    SCALE_FACTOR = 2  # This is fixed
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 50
    EPOCHS = 30 # 300
    DEVICE = "cpu"  # "cuda:0"
    DROPOUT = 0.5
    LOSS = 'l1'

class CVAE_setting:
    EMBEDDING = 128
    NUM_CHANNELS = 2
    STRIDE = 2  # This is fixed
    KERNEL_SIZE = 4  # This is fixed
    SCALE_FACTOR = 2  # This is fixed
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 50
    EPOCHS = 30  # 300
    DEVICE = "cpu"  # "cuda:0"
    DROPOUT = 0.5
    LOSS = 'l1'
    LOSS_FACTOR = 1
