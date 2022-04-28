import numpy as np
import pandas as pd
import torch
from torch.nn import BatchNorm1d, Linear, Module, Parameter, ReLU, LeakyReLU, Sequential, Tanh, MSELoss, Dropout
from torch.nn.functional import cross_entropy
from torch.nn import functional
from packaging import version

from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

### added for validation
from sklearn.model_selection import train_test_split
import optuna


################################### Defining the Neural Networks #########################################

### Define Encoder
class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim, drop):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                BatchNorm1d(item),
                LeakyReLU(0.2),
                Dropout(drop)
            ]
            dim = item
        seq.append(Linear(dim, embedding_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


######### Define Decoder
class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim,drop):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim,item), BatchNorm1d(item), LeakyReLU(0.2), Dropout(drop)]
            dim = item

        seq.append(Linear(dim, data_dim))
        seq.append(Tanh())
        self.seq = Sequential(*seq)

    def forward(self, input):
       return self.seq(input)




#################################Initialising the TAE model ##################################################
class TAESynthesizer(object):
    """TVAESynthesizer."""

    def __init__(self, WIDTH, EMBEDDING, LEARNING_RATE, EPOCHS, BATCH, DROPOUT, l2scale=1e-5,
                 trained_epoches=0,
                 log_frequency=True):
        self.WIDTH = WIDTH
        self.DROPOUT = DROPOUT
       # self.DEPTH = 2
        #self.compress_dims = np.array([int(self.WIDTH), int(self.WIDTH/2)])
        #self.decompress_dims = np.array([int(self.WIDTH/2), int(self.WIDTH)])
        self.embedding_dim = EMBEDDING
        self.compress_dims = np.array([self.WIDTH])
       # self.compress_dims  = COMPRESS
       # self.decompress_dims = DECOMPRESS
        self.decompress_dims =  np.array([self.WIDTH])
        self.log_frequency = log_frequency
        self.l2scale = l2scale
        self.batch_size = BATCH
        self.epochs = EPOCHS
        self.lr = LEARNING_RATE
        self.trained_epoches = trained_epoches
        self.total_loss = []
        self.val_loss = []


    def fit(self, data, trial=None):
       # temp_test_size = 15 / (70 + 15)  # 0.176
        exact_val_size = 0.25

        train_data, val_data = train_test_split(data.values, test_size=exact_val_size,
                                                                random_state=0)
        print(train_data.shape)

        data_dim = train_data.shape[1]
        print(data_dim)
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim, self.DROPOUT)

        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim, self.DROPOUT)

        # Initialise the optimizer
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
            weight_decay=self.l2scale)

        assert self.batch_size % 2 == 0

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        # Start training
        for i in range(self.epochs):
            self.decoder.train()  ##switch to train mode
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):
                idx = np.random.choice(np.arange(len(train_data)), self.batch_size)
                real = train_data[idx]
                optimizerAE.zero_grad()
                real = torch.from_numpy(real.astype('float32'))

                emb = self.encoder(real)
                rec = self.decoder(emb)
                L2 = MSELoss()
                loss = L2(real, rec)
                loss.backward()
                ################## 7. Updating the weights and biases using Adam Optimizer #######################################################
                optimizerAE.step()
                #self.decoder.sigma.data.clamp_(0.01, 1.0)

            ################### 8. Calculate validation loss for TVAE #############################################################
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                this_val_data = torch.from_numpy(val_data.astype('float32'))
                emb_val = self.encoder(this_val_data)
                rec_val = self.decoder(emb_val)

                L2_val = MSELoss()
                val_loss = L2_val(this_val_data, rec_val)
                self.val_loss.append(val_loss.detach().cpu())

                self.encoder.train()
                self.decoder.train()

            self.total_loss.append(loss.detach().cpu().numpy())
            print("Epoch " + str(self.trained_epoches) +
                  ", Training Loss: " + str(loss.detach().numpy()) +
                  ", Validation loss: " + str(val_loss.detach().numpy()))
            # Use Optuna for hyper-parameter tuning
            if trial is not None:
               self.optuna_metric = val_loss.detach().numpy()


               trial.report(self.optuna_metric, i)
                # Handle pruning based on the intermediate value.
               if trial.should_prune():
                   raise optuna.exceptions.TrialPruned()

    def encoded(self, data):
        self.encoder.eval()
        this_trans_data = torch.from_numpy(data.astype('float32'))
        return self.encoder(this_trans_data).detach().numpy()

    def reconstruct(self, data):
        self.encoder.eval()
        self.decoder.eval()
        this_data = torch.from_numpy(data.values.astype('float32'))
        emb_rec = self.encoder(this_data)
        rec_rec = self.decoder(emb_rec).detach().numpy()
        rec_rec = pd.DataFrame(rec_rec)
        rec_rec.columns = data.columns
        data = data.reset_index()
        rec_error = np.sum((rec_rec - data)**2,axis=1)/rec_rec.shape[1]

        return rec_rec, rec_error


    #########################################Save the model into pkl file (the whole model + training loss + validation loss) #####################
    def save(self, path):
        # always save a cpu model.
      #  device_bak = self.device
       # self.device = torch.device("cpu")
        self.encoder
        self.decoder

        torch.save(self, path)

        self.encoder
        self.decoder

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        # model.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        model.encoder
        model.de
