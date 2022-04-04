import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import BatchNorm3d, DataParallel,Conv3d, ConvTranspose3d, LeakyReLU, Module, ReLU, Sequential,Flatten, Linear, Sigmoid, Dropout, init, Tanh,functional
from torch.optim import Adam
from torchsummary import summary
from Code.config import AE_setting as cfg
from Code.logger import Logger
# NOTE: Added conditional generator to the code.

### added for validation
from sklearn.model_selection import train_test_split
import Code.Metrics as M
# import optuna

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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
            self.encoded_side = np.floor((self.encoded_side - cfg.KERNEL_SIZE) / cfg.STRIDE + 1)
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Bottleneck(Module):
    def __init__(self, bn_channel,bn_side):
        super(Bottleneck, self).__init__()
        seq = []
        seq += [Flatten(), Dropout(cfg.DROPOUT), Linear(int(bn_channel*bn_side*bn_side*bn_side), cfg.EMBEDDING)]
        seq += [Linear(cfg.EMBEDDING, int(bn_channel*bn_side*bn_side*bn_side)),Dropout(cfg.DROPOUT)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        res_z = self.seq(input)
        return res_z.reshape(list(input.shape)).float()


class Decoder(Module):
    def __init__(self, side, num_channels):
        super(Decoder, self).__init__()
        layer_dims = [(1, side), (num_channels, side // cfg.SCALE_FACTOR)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 5:  ## max of 5 for the case side = 64
            layer_dims.append(
                (layer_dims[-1][0] * cfg.SCALE_FACTOR, layer_dims[-1][1] // cfg.SCALE_FACTOR))
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
        return self.seq(input)


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

#######################4. Fretchet Inception Distance ############################################



##################5. Creating the model ################################################
class AutoEncoder(object):
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
        self.device = cfg.DEVICE  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        self.train_loss_vec = []
        self.val_loss_vec = []
        self.Loss = cfg.LOSS
        #self.optuna_metric = None


    # calculate frechet inception distance

    def fit(self, data, validation=True,model_summary=False,reload=False):

        if reload:
            self.trained_epoches = 0

        # NOTE: changed data_dim to self.data_dim. It'll be used later in sample function.
        self.data_dim = data.shape
        self.logger.write_to_file('data dimension: ' + str(self.data_dim))
        data = data.to(self.device,non_blocking=True)

        # compute side after transformation
        self.side = self.data_dim[2]
        self.depth = self.data_dim[1]
        self.logger.write_to_file('side is: ' + str(self.side))
        self.logger.write_to_file('depth is: ' + str(self.depth))
        temp_test_size = 15 / (70 + 15)
        exact_val_size = int(temp_test_size * data.shape[0])
        if validation:
            train_data, val_data = train_test_split(data, test_size=exact_val_size,
                                                    random_state=42)
        else:
            train_data = data

        self.logger.write_to_file("Training sample size: " + str(train_data.shape))
        self.logger.write_to_file("Validation sample size: " + str(val_data.shape))


        # layers_D, layers_E = determine_layers(
        #         self.side, self.random_dim, self.num_channels)

        if not reload:
            self.encoder = Encoder(self.side, self.num_channels).to(self.device)
            self.bn = Bottleneck(self.encoder.encoded_channel,self.encoder.encoded_side).to(self.device)
            self.decoder = Decoder(self.side, self.num_channels).to(self.device)

        if model_summary:
            print("*" * 100)
            print("Encoder")
            # in determine_layers, see side//2.
            summary(self.encoder)
            print("*" * 100)
            print("Bottleneck")
            summary(self.bn)
            print("*" * 100)
            print("Decoder")
            summary(self.decoder)
            print("*" * 100)


        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.bn.parameters()) + list(self.decoder.parameters()) , lr=self.lr,
            weight_decay=self.l2scale)

        #assert self.batch_size % 2 == 0
###################10. Start training ############################################################
        for i in range(self.epochs):
            self.decoder.train()  ##switch to train mode
            self.encoder.train()
            self.bn.train()
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
                encoded_data = self.encoder(batch)
                bn_output = self.bn(encoded_data)
                recon_x = self.decoder(bn_output)

                ################### 6. Calculate training loss #################################################################
                if self.device == 'cpu':
                    loss = M.loss_fun(batch, recon_x)
                else:
                    loss = M.loss_fun(batch.cpu(),recon_x.cpu())

                loss.backward()
                ################## 7. Updating the weights and biases using Adam Optimizer #######################################################
                optimizerAE.step()
            self.train_loss_vec.append(loss.detach().numpy())
            self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                                      ", Training Loss: " + str(loss.detach().numpy()))
            if validation:
                recon_val_data = self.reconstruct(val_data)
                if self.device == 'cpu':
                    val_loss = M.loss_fun(val_data,recon_val_data)
                else:
                    val_loss = M.loss_fun(val_data.cpu(), recon_val_data.cpu())
                self.val_loss_vec.append(val_loss.detach().numpy())
                self.logger.write_to_file("Epoch " + str(self.trained_epoches) +
                      ", Validation Loss: " + str(val_loss.detach().numpy()))

    def reconstruct(self,data):
        self.encoder.eval()  ## switch to evaluate mode
        self.decoder.eval()
        self.bn.eval()
        encoded_data = self.encoder(data.unsqueeze(1).to(self.device))
        decoder_input = self.bn(encoded_data)
        output = self.decoder(decoder_input).squeeze(1)
        return output

########################################Save the model into pkl file (the whole model + training loss + validation loss) #####################
    def save(self, path):
        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.bn.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.bn.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device(cfg.DEVICE)  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        model.encoder.to(model.device)
        model.decoder.to(model.device)
        model.bn.to(model.device)

        return model
