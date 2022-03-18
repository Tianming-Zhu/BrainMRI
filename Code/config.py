# centralized configuration file.
# It can be updated using argparse as well.



class ctgan_setting:
    EMBEDDING = 128
    DEPTH = 2
    WIDTH = 256
    GENERATOR_LEARNING_RATE = 2e-4
    DISCRIMINATOR_LEARNING_RATE = 2e-5
    BATCH_SIZE = 500
    EPOCHS = 2 # 800
    DROPOUT = 0.5
    DISCRIMINATOR_STEP = 1
    CONDGEN = True
    DEVICE = "cpu"  # "cuda:0"


class tvae_setting:
    EMBEDDING = 128
    DEPTH = 2
    WIDTH = 128
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 500
    EPOCHS = 20 # 300
    CONDGEN_ENCODER = True
    CONDGEN_LATENT = True
    DEVICE = "cpu"  # "cuda:0"


class AE_setting:
    EMBEDDING = 128
    NUM_CHANNELS = 2
    STRIDE = 2  # This is fixed
    KERNEL_SIZE = 4  # This is fixed
    SCALE_FACTOR = 2  # This is fixed
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 50
    EPOCHS = 2 # 300
    DISCRIMINATOR_STEP = 1
    DEVICE = "cpu"  # "cuda:0"
    DROPOUT = 0.5
 
       
     