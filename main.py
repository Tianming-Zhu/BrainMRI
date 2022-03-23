import numpy as np
import pandas as pd
import math
import nibabel as nib
import torch
import glob
import os
import time
from sklearn.model_selection import train_test_split
import sys
from Code.argparser import ParserOutput
from Code.logger import Logger
import optuna
from Code import config as cfg
sys.path.append('/home/stazt/BrainMRI')
from Code import Autoencoder
filelist = glob.glob('/home/stazt/Public_Dataset/sub*')
images = []
for i in np.arange(len(filelist)):
    img = nib.load(filelist[i])
    images.append(img.get_fdata())
data = torch.from_numpy(np.array(images)).float()
train_val_data, testdata = train_test_split(data, test_size=0.15, random_state=42)


'''
Run parser function to update inputs by the user.
'''
# update inputs
parser = ParserOutput()

# GLOBAL PARAMETERS
# These parameters are needed in Optuna training and have to be placed outside of the functions.
# model to be trained
model = None
# to record the top num_max_mdls models
metric_vals = []
mdl_fns = []
num_max_mdls = parser.max_num_mdls

# def get_train_data():
#     # get paths
#     data_path = os.path.join(parser.datadir, parser.data_fn)
#     discrete_cols_path = os.path.join(parser.datadir, parser.discrete_fn)
#
#     if not os.path.isfile(data_path):
#         ValueError('Training data file ' + data_path + " does not exists.")
#
#     if not os.path.isfile(discrete_cols_path):
#         ValueError('Discrete text file ' + discrete_cols_path + " does not exists.")
#
#     if parser.transformer is None:
#         data = pd.read_csv(data_path)
#     else:
#         # if transformer is provided, then the data should have been transformed
#         # moreover, the saved data is in a numpy array.
#         data = pd.read_csv(data_path).values
#
#     # read list of discrete variables
#     with open(discrete_cols_path, "r+") as f:
#         discrete_columns = f.read().splitlines()
#
#     return data, discrete_columns


def run_individual_training():
    # Initialize seed
    torch.manual_seed(parser.torch_seed)
    np.random.seed(parser.numpy_seed)

    #select model
    if parser.model_type == 'ae':
        model = Autoencoder.AutoEncoder()
    elif parser.model_type == 'CVAE':
        model = CVAE.VAutoEncoder()
    else:
        ValueError('The selected model, ' + parser.model_type + ', is invalid.')

   # Create a new folder (with PID) to save the training results
    model.logger.change_dirpath(model.logger.dirpath + "/" + parser.model_type + "_" + model.logger.PID)

    # Record the seed numbers
    model.logger.write_to_file('PyTorch seed number ' + str(parser.torch_seed))
    model.logger.write_to_file('Numpy seed number ' + str(parser.numpy_seed))

    # update logger output path
    model.logger.change_dirpath(parser.outputdir)

    # Train the model
    start_time = time.time()
    model.fit(train_val_data, validation=True,model_summary=False)
    elapsed_time = time.time() - start_time
    model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)

    # Use same PID as log file for sample csv and the trained model pkl files.
    PIDTAG = model.logger.PID + "_" + model.logger.dt.now().strftime(model.logger.datetimeformat)

    # Save the model in the same folder as the pkl file
    model_fn = parser.model_type + "_" + PIDTAG + ".pkl"
    output_model_path = os.path.join(model.logger.dirpath, model_fn)
    model.save(output_model_path)


def run_optuna_training():
    # function to sort two lists together
    def sortlists(metrics, fns):
        metrics_sorted, fns_sorted = (list(t) for t in zip(*sorted(zip(metrics, fns))))
        return metrics_sorted, fns_sorted

    # logger to save optuna statistics
    optuna_logger = Logger(filename="optuna_trials_summary.txt")
    optuna_logger.change_dirpath(parser.outputdir + "/" + parser.model_type + "_" + optuna_logger.PID)

    # generate unique seed for each trial
    seed_list = np.arange(parser.trials).tolist()
    np.random.shuffle(seed_list)

    def objective(trial):
        global model

        # get the seed number from the last element of seed_list and remove it from the list.
        this_seed = seed_list[-1]
        seed_list.pop()
        torch.manual_seed(this_seed)
        np.random.seed(this_seed)

        # NOTE: uncomment the cfg.(respective hyper-parameters) if we want Optuna to vary more variables.
        if parser.model_type == 'ctgan':
            cfg.ctgan_setting.GENERATOR_LEARNING_RATE = trial.suggest_categorical('ct_gen_lr', [1e-6, 2e-6, 1e-5, 2e-5])
            #cfg.ctgan_setting.DISCRIMINATOR_LEARNING_RATE = trial.suggest_categorical('ct_dis_lr', [1e-6, 2e-6, 1e-5, 2e-5])
            #cfg.ctgan_setting.EPOCHS = trial.suggest_int('ct_epochs', 600, 900, step=100)
            # cfg.ctgan_setting.BATCH_SIZE = trial.suggest_int('ct_batchsize', 500, 1000, step=100)
            # cfg.ctgan_setting.DEPTH = trial.suggest_int('ct_depth', 1, 3)
            # cfg.ctgan_setting.WIDTH = trial.suggest_int('ct_width', 128, 512, step=64)
            # # cfg.ctgan_setting.EMBEDDING = trial.suggest_int('ct_embedding', 128, 512, step=64)
            # cfg.ctgan_setting.DROPOUT = trial.suggest_categorical('ct_dropout', [0.25, 0.5])

            model = CTGANSynthesizer()  # initialize a new model

        elif parser.model_type == 'tablegan':
            cfg.tablegan_setting.LEARNING_RATE = trial.suggest_categorical('tbl_lr', [2e-6, 5e-6, 1e-5])
            # cfg.tablegan_setting.BATCH_SIZE = trial.suggest_int('tbl_batchsize', 500, 800, step=100)
            # cfg.tablegan_setting.EPOCHS = trial.suggest_int('tbl_epochs', 150, 300, step=50)

            model = TableganSynthesizer()  # initialize a new model

        elif parser.model_type == 'tvae':
            cfg.tvae_setting.LEARNING_RATE = trial.suggest_categorical('tv_lr', [1e-5, 1e-4, 1e-3])  # 1e-2 results in non-decreasing loss
            # cfg.tvae_setting.EPOCHS = trial.suggest_int('tv_epochs', 300, 900, step=100)
            # cfg.tvae_setting.BATCH_SIZE = trial.suggest_int('tv_batchsize', 500, 1000, step=100)
            # cfg.tvae_setting.DEPTH = trial.suggest_int('tv_depth', 1, 4)
            # cfg.tvae_setting.WIDTH = trial.suggest_int('tv_width', 128, 512, step=64)
            # cfg.tvae_setting.EMBEDDING = trial.suggest_int('tv_embedding', 128, 512, step=64)
            # # cfg.tvae_setting.CONDGEN = trial.suggest_categorical('tv_condgen', [True, False])
            # cfg.tvae_setting.CONDGEN_ENCODER = trial.suggest_categorical('tv_condgen_encoder', [True, False])
            # cfg.tvae_setting.CONDGEN_LATENT = trial.suggest_categorical('tv_condgen_latent', [True, False])

            model = TVAESynthesizer() # initialize a new model

        else:
            ValueError('The selected model, ' + parser.model_type + ', is invalid.')

        # Create a new folder (with PID) to save the training results
        model.logger.change_dirpath(optuna_logger.dirpath)

        # Add trial number to summary log file
        filename, file_extension = os.path.splitext(model.logger.filename)
        new_filename = filename + "_" + str(trial.number) + file_extension
        model.logger.change_filename(new_filename)

        # Record the seed number
        model.logger.write_to_file('PyTorch seed number ' + str(this_seed))
        model.logger.write_to_file('Numpy seed number ' + str(this_seed))

        data, discrete_columns = get_train_data()

        # Train the model
        start_time = time.time()

        # model.fit(data, discrete_columns)
        model.fit(data, discrete_columns, model_summary=False, trans="VGM", trial=trial,
                  transformer=parser.transformer, in_val_data=parser.val_data)

        elapsed_time = time.time() - start_time
        model.logger.write_to_file("Training time {:.2f} seconds".format(elapsed_time), True)

        return model.optuna_metric

################################At the end of the optuna trial, save the model ################################
    def callback(study, trial):
        global metric_vals, mdl_fns, num_max_mdls  # , best_mdl
        this_model_fn = parser.model_type + "_model_" \
                        + str(trial.number) + "_" \
                        + optuna_logger.PID + "_" \
                        + optuna_logger.dt.now().strftime(optuna_logger.datetimeformat)

        if trial.state == optuna.trial.TrialState.COMPLETE:
            if len(metric_vals) < num_max_mdls:
                metric_vals.append(model.optuna_metric)
                mdl_fns.append(this_model_fn)
                metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")
            else:
                if model.optuna_metric < metric_vals[-1]:
                    # remove the previously saved model
                    metric_vals.pop()
                    mdl_fn_discard = mdl_fns.pop()

                    os.remove(optuna_logger.dirpath + "/" + mdl_fn_discard + ".pkl")

                    # add the new record
                    metric_vals.append(model.optuna_metric)
                    mdl_fns.append(this_model_fn)
                    metric_vals, mdl_fns = sortlists(metric_vals, mdl_fns)

                    model.save(optuna_logger.dirpath + "/" + this_model_fn + ".pkl")

        # write intermediate Optuna training results, in case the training crashes, eg. core dumped error.
        optuna_logger.write_to_file("Trial: {}, Metric: {}, Status: {}, model fn: {} ".format(
            trial.number, model.optuna_metric, trial.state, this_model_fn), toprint=False)

    # NOTE: We can choose different methods to sample the hyper-parameters.
    # Training with TPE multivariate=True is reported to give better results than default TPE
    # See https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/
    '''
    ## Use GridSampler
    if parser.model_type == 'ctgan':
        search_space = {"ct_gen_lr": [1e-5, 2e-5], "ct_width": [384, 512], "ct_batchsize": [600, 700]}
    elif parser.model_type == 'tablegan':
        search_space = {"tbl_lr": [1e-5, 5e-6], "tbl_epochs": [150, 200], "tbl_batchsize": [500, 800]}
    elif parser.model_type == 'tvae':
        search_space = {"tv_lr": [1e-5, 1e-4], "tv_epochs": [300, 400], "tv_batchsize": [500, 600]}
    sampler = optuna.samplers.GridSampler(search_space)
    '''

    # Use TPESampler that uses Bayesian Optimization to constrict the range of hyper-parameters
    sampler = optuna.samplers.TPESampler(multivariate=True)

    if parser.pruner:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5,
                                             n_warmup_steps=parser.warmup_steps,
                                             interval_steps=10)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    pruner=optuna.pruners.NopPruner())

    # Remove/replace NopPruner if we want to use a pruner.
    # See https://optuna.readthedocs.io/en/v1.4.0/reference/pruners.html
    # study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=parser.trials, callbacks=[callback])
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Training completed. Clear the intermediate log that was saved by optuna_logger first.
    optuna_logger.clear_all_content()

    # write training statistics to log file
    optuna_logger.write_to_file("Study statistics: ")
    optuna_logger.write_to_file("  Number of finished trials: " + str(len(study.trials)))
    optuna_logger.write_to_file("  Number of pruned trials: " + str(len(pruned_trials)))
    optuna_logger.write_to_file("  Number of complete trials: " + str(len(complete_trials)))

    optuna_logger.write_to_file("Best trial:")
    trial = study.best_trial

    optuna_logger.write_to_file("  Value: " + str(trial.value))

    optuna_logger.write_to_file("  Params: ")
    for key, value in trial.params.items():
        optuna_logger.write_to_file("    {}: {}".format(key, value))

    optuna_logger.write_to_file(study.trials_dataframe().sort_values(by='value', ascending=True).to_string(header=True, index=False))

    # Generate synthetic data with each save model
    for mdl_fn in mdl_fns:
        filepath = optuna_logger.dirpath + "/" + mdl_fn + ".pkl"
        sample_fn = mdl_fn + "_sample.csv"
        output_sample_path = os.path.join(optuna_logger.dirpath, sample_fn)
        temp_mdl = torch.load(filepath, map_location=torch.device('cpu'))
        samples = temp_mdl.sample(parser.samplesize, condition_column=None, condition_value=None)
        samples.to_csv(output_sample_path, index=False, header=True)


if __name__ == "__main__":
    if parser.proceed:
        if parser.use_optuna:
            run_optuna_training()
        else:
            run_individual_training()
