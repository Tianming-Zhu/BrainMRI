import argparse
import os
from Code import config as cfg
import pandas as pd

# To allow True/False argparse input.
# See answer by Maxim in https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_args():
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument("-nv", "--nv", help="run nvidia-smi command", action="store_true")
    parser.add_argument("--torch_seed", default=0, type=int, metavar='', help="PyTorch random seed")
    parser.add_argument("--numpy_seed", default=0, type=int, metavar='', help="PyTorch random seed")
    parser.add_argument('--model', default=None, type=str, metavar='', help='ctgan, tablegan or tvae')
    parser.add_argument('--datadir', default=None, type=str, metavar='', help='path of training data directory')
    parser.add_argument('--outputdir', default=None, type=str, metavar='', help='path of output directory')
    #parser.add_argument('--data_fn', default=None, type=str, metavar='', help='filename of transformed training data (with .csv)')
    #parser.add_argument('--val_data_fn', default=None, type=str, metavar='', help='filename of validation data (with .csv)')
    # Optuna
    parser.add_argument('--use_optuna', default=False, type=str2bool, metavar='', help='set True to use Optuna')
    parser.add_argument('--trials', default=10, type=int, metavar='', help='Number of Optuna trials')
    parser.add_argument('--max_num_mdls', default=5, type=int, metavar='',  help='Number of Optuna trials')
    parser.add_argument('--pruner', default=True, type=str2bool, metavar='', help='use Optuna pruner')
    parser.add_argument('--warmup_steps', default=0, type=int, metavar='', help='Optuna median pruner warm-up')

    # Autoencoder parameters
    parser.add_argument('--AE_embedding', default=None, type=int, metavar='', help='Autoencoder embedding')
    parser.add_argument('--AE_num_channels', default=None, type=int, metavar='',
                        help='Autoencoder number of channels')
    parser.add_argument('--AE_stride', default=None, type=float, metavar='', help='Autoencoder stride')
    parser.add_argument('--AE_kernel_size', default=None, type=float, metavar='',
                        help='Autoencoder kernel size')
    parser.add_argument('--AE_scale_factor', default=None, type=float, metavar='',
                        help='Autoencoder scale factor')
    parser.add_argument('--AE_lr', default=None, type=float, metavar='', help='Autoencoder learning rate')
    parser.add_argument('--AE_batchsize', default=None, type=int, metavar='', help='Autoencoder batch size')
    parser.add_argument('--AE_epochs', default=None, type=int, metavar='', help='Autoencoder num epochs')
    parser.add_argument('--AE_dropout', default=None, type=float, metavar='', help='Autoencoder dropout rate')
    parser.add_argument('--AE_dis_step', default=None, type=int, metavar='', help='Autoencoder discriminator step')
    parser.add_argument('--AE_device', default=None, type=str, metavar='', help='Autoencoder cpu or cuda')
    parser.add_argument('--AE_metric', default=None, type=str, metavar='', help='Metric used for measuring reconstructed error')



    # CVAE parameters
    parser.add_argument('--tv_embedding', default=None, type=int, metavar='', help='tvae embedding')
    # parser.add_argument('--tv_condgen', default=None, type=str2bool, metavar='', help='tvae cond. gen.')
    parser.add_argument('--tv_depth', default=None, type=int, metavar='', help='tvae num hidden layers')
    parser.add_argument('--tv_width', default=None, type=int, metavar='', help='tvae width of mlp')
    parser.add_argument('--tv_lr', default=None, type=float, metavar='', help='tvae learning rate')
    parser.add_argument('--tv_batchsize', default=None, type=int, metavar='', help='tvae batch size')
    parser.add_argument('--tv_epochs', default=None, type=int, metavar='', help='tvae num epochs')
    parser.add_argument('--tv_device', default=None, type=str, metavar='', help='tvae cpu or cuda')
    parser.add_argument('--tv_condgen_encoder', default=None, type=str2bool, metavar='', help='tvae cond. gen. to encoder')
    parser.add_argument('--tv_condgen_latent', default=None, type=str2bool, metavar='', help='tvae cond. gen. to latent space')

    return parser.parse_args()


class ParserOutput:
    """
        To store output values in an object instead of returning individual values
        The default datadir and outputdir are set to /workspace for ease of using Docker container.
    """
    def __init__(self):
        self.proceed = False
        self.torch_seed = 0
        self.numpy_seed = 0
        self.model_type = None
        self.datadir = None
        self.outputdir = None
       # self.data_fn = None
       # self.discrete_fn = None
       # self.val_data = None
       # self.val_transformed_data = None

        self.parser_func()

    def parser_func(self):
        """
        The function acts as a placeholder to update cfg with values from argparse.

        NOTE:
            proceed: continue with subsequent code in main.py
            model: the selected model is either ctgan, tablegan or tvae.
            datadir: where the training data is located.
            outputdir: where the trained model should be stored.
            data_fn: file nAEame of training data.
            discrete_fn: file that contains the names of discrete variables.
        """

        args = _parse_args()

        # enable user to run the nvidia-smi command
        # without having to run Docker container in interactive mode.
        if args.nv:
            os.system("nvidia-smi")
            return

        # sanity check
        if args.model is None:
            print('Please specify --model.')
            return

        # if args.data_fn is None:
        #     print('Please specify --data_fn.')
        #     return


        # Store values
        self.torch_seed = args.torch_seed
        self.numpy_seed = args.numpy_seed
        self.model_type = args.model.lower()
        self.datadir = args.datadir
        self.outputdir = args.outputdir
      #  self.data_fn = args.data_fn
       # self.val_data_fn = args.val_data_fn


        # Optuna
        self.use_optuna = args.use_optuna
        self.trials = args.trials
        self.max_num_mdls = args.max_num_mdls
        self.pruner = args.pruner
        self.warmup_steps = args.warmup_steps


        # if args.val_data_fn is not None:
        #     self.val_data = pd.read_csv(os.path.join(self.datadir, args.val_data_fn))


        if self.model_type == 'ae':
            if args.AE_embedding is not None:
                cfg.AE_setting.EMBEDDING = args.AE_embedding

            if args.AE_num_channels is not None:
                cfg.AE_setting.NUM_CHANNELS = args.AE_num_channels

            if args.AE_stride is not None:
                cfg.AE_setting.STRIDE = args.AE_stride

            if args.AE_kernel_size is not None:
                cfg.AE_setting.KERNEL_SIZE = args.AE_kernel_size

            if args.AE_scale_factor is not None:
                cfg.AE_setting.SCALE_FACTOR = args.AE_scale_factor

            if args.AE_lr is not None:
                cfg.AE_setting.LEARNING_RATE = args.AE_lr

            if args.AE_batchsize is not None:
                cfg.AE_setting.BATCH_SIZE = args.AE_batchsize

            if args.AE_epochs is not None:
                cfg.AE_setting.EPOCHS = args.AE_epochs

            if args.AE_dropout is not None:
                cfg.AE_setting.DROPOUT = args.AE_dropout

            if args.AE_dis_step is not None:
                cfg.AE_setting.DISCRIMINATOR_STEP = args.AE_dis_step

            if args.AE_device is not None:
                cfg.AE_setting.DEVICE = args.AE_device

            if args.AE_metric is not None:
                cfg.AE_setting.LOSS = args.AE_metric




        elif self.model_type == 'tablegan':

            if args.tbl_embedding is not None:
                cfg.tablegan_setting.EMBEDDING = args.tbl_embedding

            if args.tbl_num_channels is not None:
                cfg.tablegan_setting.NUM_CHANNELS = args.tbl_num_channels

            if args.tbl_dlayer is not None:
                cfg.tablegan_setting.DLAYER = args.tbl_dlayer

            if args.tbl_lr is not None:
                cfg.tablegan_setting.LEARNING_RATE = args.tbl_lr

            if args.tbl_batchsize is not None:
                cfg.tablegan_setting.BATCH_SIZE = args.tbl_batchsize

            if args.tbl_epochs is not None:
                cfg.tablegan_setting.EPOCHS = args.tbl_epochs

            if args.tbl_dis_step is not None:
                cfg.tablegan_setting.DISCRIMINATOR_STEP = args.tbl_dis_step

            if args.tbl_device is not None:
                cfg.tablegan_setting.DEVICE = args.tbl_device

        else:
            print('Please specify the correct model type.')
            return

        self.proceed = True
