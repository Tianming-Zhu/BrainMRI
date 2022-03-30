import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss, BatchNorm3d, Conv3d, ConvTranspose3d, LeakyReLU, MSELoss, Module, ReLU, Sequential,Flatten, Linear, Sigmoid, Dropout, init, Tanh,functional
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torchsummary import summary

from Code.config import CVAE_setting as cfg
from Code.logger import Logger
# NOTE: Added conditional generator to the code.

### added for validation
from sklearn.model_selection import train_test_split
import Code.Metrics as M
# import optuna

################################# 1. Define our neural networks ###################################################
class Encoder(Module):
    def __init__(self, side, num_channels):
        super(Encoder, self).__init__()

        layer_dims = [(1, side), (num_channels, side // cfg.SCALE_FACTOR)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 5:  ## max of 5 for the case side = 64
            layer_dims.append(
                (layer_dims[-1][0] * cfg.SCALE_FACTOR, layer_dims[-1][1] // cfg.SCALE_FACTOR))
        # layers of Decoder
        self.encoded_side = side
        seq = []
        for prev, curr in zip(layer_dims, layer_dims[1:]):
            seq += [
                # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                Conv3d(prev[0], curr[0], cfg.KERNEL_SIZE, cfg.STRIDE,  padding=0, bias=False),
                BatchNorm3d(curr[0]),
                ## the slope of the leak was set to 0.2
               # LeakyReLU(0.2, inplace=True)  ##y=0.2x when x<0
                ReLU(True)
            ]
            self.encoded_channel = curr[0]
            self.encoded_side = np.floor((self.encoded_side - cfg.KERNEL_SIZE) / cfg.STRIDE + 1).astype('int')
        self.seq = Sequential(*seq)
        fc1 = [Flatten(), Dropout(cfg.DROPOUT), Linear(int(self.encoded_channel * self.encoded_side *
                                                                self.encoded_side * self.encoded_side), cfg.EMBEDDING)]
        fc2 = [Flatten(), Dropout(cfg.DROPOUT), Linear(int(self.encoded_channel * self.encoded_side *
                                                                self.encoded_side * self.encoded_side), cfg.EMBEDDING)]
        self.fc1 = Sequential(*fc1)
        self.fc2 = Sequential(*fc2)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar



class Decoder(Module):
    def __init__(self, side, num_channels,input_channel,input_side):
        super(Decoder, self).__init__()
        layer_dims = [(1, side), (num_channels, side // cfg.SCALE_FACTOR)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 5:  ## max of 5 for the case side = 64
            layer_dims.append(
                (layer_dims[-1][0] * cfg.SCALE_FACTOR, layer_dims[-1][1] // cfg.SCALE_FACTOR))
        self.input_channel = input_channel
        self.input_side = input_side
        fc3 = [Linear(cfg.EMBEDDING, int(self.input_channel*self.input_side
                                         *self.input_side * self.input_side)),Dropout(cfg.DROPOUT)]
        self.fc3 = Sequential(*fc3)
        seq = []
        for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
            seq += [
                BatchNorm3d(prev[0]),
                #LeakyReLU(0.2, inplace=True),
                ReLU(True),
                ConvTranspose3d(prev[0], curr[0], cfg.KERNEL_SIZE, cfg.STRIDE, output_padding=0, bias=False)
            ]
        seq += [ReLU(True)]
        self.seq = Sequential(*seq)
    def forward(self, input):
        res_input = self.fc3(input)
        output = self.seq(res_input.reshape([input.shape[0],self.input_channel, self.input_side,
                                                self.input_side, self.input_side])).float()
        return output

def loss_function(recon_x, x, mu, logvar, factor):
    ## This loss equals KLD + reconstruction error
    loss = M.loss_fun(x, recon_x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return average loss per batch
    return loss * factor, KLD/(mu.size()[0]*mu.size()[1])

###########################3. Initialise weights ################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        ###All weights for convolutional and de-convolutional layers were initialized
        # from a zero-centered Normal distribution with standard deviation 0.02.
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)



##################5. Creating the model ################################################
class CVAutoEncoder(object):
    """docstring for TableganSynthesizer"""

    def __init__(self, l2scale=1e-5, trained_epoches = 0, log_frequency=True):

        self.random_dim = cfg.EMBEDDING
        self.num_channels = cfg.NUM_CHANNELS
        self.l2scale = l2scale
        self.epochs = cfg.EPOCHS
        self.lr = cfg.LEARNING_RATE
        self.log_frequency = log_frequency
        self.batch_size = cfg.BATCH_SIZE
        self.trained_epoches = trained_epoches
        self.side = 0
        self.data_dim = 0
        self.logger = Logger()
        self.loss_factor = cfg.LOSS_FACTOR
        self.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        self.train_loss_vec = []
        self.val_loss_vec = []
        #self.optuna_metric = None




    def fit(self, data, validation=True,model_summary=False,reload=False):
        if reload:
            self.trained_epoches = 0

        # NOTE: changed data_dim to self.data_dim. It'll be used later in sample function.
        self.data_dim = data.shape
        print('data dimension: ' + str(self.data_dim))
        data = data.to(self.device, non_blocking=True)

        # compute side after transformation
        self.side = self.data_dim[2]
        self.depth = self.data_dim[1]
        print('side is: ' + str(self.side))
        print('depth is: ' + str(self.depth))
        temp_test_size = 15 / (70 + 15)
        exact_val_size = int(temp_test_size * data.shape[0])
        if validation:
            train_data, val_data = train_test_split(data, test_size=exact_val_size,
                                                    random_state=42)
        else:
            train_data = data

        # layers_D, layers_E = determine_layers(
        #         self.side, self.random_dim, self.num_channels)

        if not reload:
            self.encoder = Encoder(self.side, self.num_channels).to(self.device)
            self.decoder = Decoder(self.side, self.num_channels,self.encoder.encoded_channel,self.encoder.encoded_side).to(self.device)

        if model_summary:
            print("*" * 100)
            print("Encoder")
            # in determine_layers, see side//2.
            summary(self.encoder)
            print("*" * 100)
            print("Decoder")
            summary(self.decoder)
            print("*" * 100)


        optimizerAE = Adam(
            list(self.encoder.parameters()) +  list(self.decoder.parameters()) , lr=self.lr,
            weight_decay=self.l2scale)

        #assert self.batch_size % 2 == 0


###################10. Start training ############################################################
        for i in range(self.epochs):
            self.decoder.train()  ##switch to train mode
            self.encoder.train()
            self.trained_epoches += 1

            data_train = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
            )

            for batch_idx, samples in enumerate(data_train):
                optimizerAE.zero_grad()
                batch = samples.unsqueeze(1)
                mu, std, logvar = self.encoder(batch)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec = self.decoder(emb)

                ################### 6. Calculate training loss #################################################################
                ################### 6. Calculate training loss #################################################################
                if self.device == 'cpu':
                    loss_1, loss_2 = loss_function(
                        rec, batch, mu, logvar, self.loss_factor)
                else:
                    loss_1, loss_2 = loss_function(
                        rec.cpu(), batch.cpu(), mu.cpu(), logvar.cpu(), self.loss_factor)

                loss = loss_1 + loss_2
                self.train_loss_vec.append(loss.detach().numpy())
                loss.backward()

                ################## 7. Updating the weights and biases using Adam Optimizer #######################################################
                optimizerAE.step()

            print("Epoch " + str(self.trained_epoches) +
                  ", Training Loss: " + str(loss.detach().numpy()))
            if validation:
                recon_val_data = self.sample(val_data.shape[0])
                if self.device == 'cpu':
                    val_loss = M.loss_fun(val_data, recon_val_data)
                else:
                    val_loss = M.loss_fun(val_data.cpu(), recon_val_data.cpu())
                self.val_loss_vec.append(val_loss.detach().numpy())
                print("Epoch " + str(self.trained_epoches) +
                      ", Validation Loss: " + str(val_loss.detach().numpy()))




    def sample(self, samples):
        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            # print("ema_mu, ema_std", self.ema_mu, self.ema_std)
            # NOTE: empirically, ema_mu and ema_std are close to 0 and 1 respectively.
            # justifying the original assumptions
            mean = torch.zeros(self.batch_size, self.random_dim)
            std = mean + 1

            fakez = torch.normal(mean=mean, std=std).to(self.device)
            fake = self.decoder(fakez)
            data.append(fake)
        data = torch.cat(data, axis=0)
        data = data[:samples]
        return data.squeeze(1)

########################################Save the model into pkl file (the whole model + training loss + validation loss) #####################
    def save(self, path):
        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        model.encoder.to(model.device)
        model.decoder.to(model.device)

        return model
