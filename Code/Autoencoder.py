import numpy as np
import torch
from torch.nn import L1Loss, BatchNorm3d, Conv3d, ConvTranspose3d, LeakyReLU, MSELoss, Module, ReLU, Sequential,Flatten, Linear, Sigmoid, Dropout, init, Tanh,functional
from torch.nn.functional import binary_cross_entropy_with_logits,adaptive_avg_pool2d
from torch.optim import Adam
from scipy import linalg
from torchsummary import summary
#from pytorch_fid.inception import InceptionV3
from numpy import iscomplexobj,asarray
from scipy.linalg import sqrtm
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.inception_v3 import preprocess_input
#from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize

#from pytorch_ssim_3D.pytorch_ssim import SSIM3D

from Code.config import AE_setting as cfg
from Code.logger import Logger
# NOTE: Added conditional generator to the code.

### added for validation
from sklearn.model_selection import train_test_split
# import ctgan.metric as M
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

## scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
            ## resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
            ## store
        images_list.append(new_image)
    return asarray(images_list)

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # NOTE: original implementation "cuda:0" if torch.cuda.is_available() else "cpu"
        self.train_loss_vec = []
        self.val_loss_vec = []
        #self.optuna_metric = None


    # calculate frechet inception distance
    def calculate_fid(self,model, images1, images2):
        if torch.cuda.is_available():
          images1 = images1.cpu().detach().numpy()
          images2 = images2.cpu().detach().numpy()
        else:
          images1 = images1.detach().numpy()
          images2 = images2.detach().numpy()
            
        # calculate activations
        images1 = scale_images(images1, (299, 299, 3))
        images2 = scale_images(images2, (299, 299, 3))

        act1 = torch.from_numpy(model.predict(images1))
        act2 = torch.from_numpy(model.predict(images2))

        mu1, sigma1 = torch.mean(act1, dim=0), torch.cov(act1.T)
        mu2, sigma2 = torch.mean(act2, dim=0), torch.cov(act2.T)
        # calculate sum squared difference between means
        ssdiff = torch.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(torch.matmul(sigma1, sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def fit(self, data, validation=True,model_summary=False,reload=False):
        #print("Device is: "+ self.device)
        if reload:
            self.trained_epoches = 0

        # NOTE: changed data_dim to self.data_dim. It'll be used later in sample function.
        self.data_dim = data.shape
        print('data dimension: ' + str(self.data_dim))
        data = data.to(self.device)

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
        print("Training sample size: " + str(train_data.shape))
        print("Validation sample size: " + str(val_data.shape))

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

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        #inception_model = keras.models.load_model('/content/BrainMRI/ctgan/inception.h5')
        #inception_model = keras.models.load_model('/home/stazt/BrainMRI/ctgan/inception.h5')
###################10. Start training ############################################################
        for i in range(self.epochs):
            self.decoder.train()  ##switch to train mode
            self.encoder.train()
            self.bn.train()
            self.trained_epoches += 1
            for id_ in range(steps_per_epoch):
                optimizerAE.zero_grad()
                #emb = torch.from_numpy(train_data.astype('float32')).to(self.device)
                indices = torch.randperm(len(train_data),device=self.device)[:self.batch_size]
                batch = train_data[indices].unsqueeze(1)
                #print(batch.squeeze(1).detach().numpy().shape)
                encoded_data= self.encoder(batch)
                bn_output = self.bn(encoded_data)
                recon_x = self.decoder(bn_output)
                #print(recon_x.squeeze(1).detach().numpy().shape)

                ################### 6. Calculate training loss #################################################################
                #loss = np.sum(np.abs(batch.detach().numpy() -recon_x.detach().numpy()))
                # L2 = MSELoss()
                # loss = L2(batch, recon_x)

                # L1 = L1Loss()
                # loss = L1(batch,recon_x)
                ## SSIM
                #ssim_loss = SSIM3D()
                #criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
                #loss = 1-ssim_loss(batch,recon_x)
                # ssim_loss = pytorch_ssim.SSIM()
               # loss = -ssim_loss(batch,recon_x)

                loss = self.calculate_fid(inception_model,batch.squeeze(1),recon_x.squeeze(1))
                #print("FID loss is: " + str(loss))
                self.train_loss_vec.append(loss.detach().numpy())
                loss.requires_grad = True
                loss.backward()
                ################## 7. Updating the weights and biases using Adam Optimizer #######################################################
                optimizerAE.step()
            print("Epoch " + str(self.trained_epoches) +
                                      ", Training Loss: " + str(loss.detach().numpy()))
            if validation:
                recon_val_data = self.reconstruct(val_data)
                #val_loss = (recon_val_data - val_data).abs().mean()
                val_loss = self.calculate_fid(inception_model,val_data.squeeze(1)),recon_val_data.squeeze(1))
                #print(val_data.squeeze(1).detach().numpy().shape)
                #print(recon_val_data.squeeze(1).detach().numpy().shape)
                self.val_loss_vec.append(val_loss.detach().numpy())
                print("Epoch " + str(self.trained_epoches) +
                      ", Validation Loss: " + str(val_loss.detach().numpy()))

    def reconstruct(self,data):
        self.encoder.eval()  ## switch to evaluate mode
        self.decoder.eval()
        self.bn.eval()
        encoded_data = self.encoder(data.unsqueeze(1))
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
