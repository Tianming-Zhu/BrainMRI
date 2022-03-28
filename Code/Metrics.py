import torch
import numpy as np
from torch.nn import L1Loss, MSELoss
from Code.config import AE_setting as cfg
from pytorch_ssim_3D.pytorch_ssim import SSIM3D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy.linalg import sqrtm

def loss_fun(real, fake):
    if cfg.LOSS == 'L1':
        L1 = L1Loss()
        return L1(real,fake)
    if cfg.LOSS == 'L2':
        L2 = MSELoss()
        return L2(real, fake)
    if cfg.LOSS == 'SSIM':
        ssim_loss = SSIM3D()
        return 1-ssim_loss(real, fake)
    if cfg.LOSS == 'FID':
        inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        loss = calculate_fid(inception_model, real, fake)
        loss.requires_grad = True
        return loss
    else:
        print("This metric is not available.")


## scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        for image_slice in image:
            ## resize with nearest neighbor interpolation
            new_image = resize(image_slice, new_shape, 0)
            ## store
            images_list.append(new_image)
    return asarray(images_list)

def calculate_fid(model, images1, images2, eps=1e-6):
    images1 = images1.cpu().detach().numpy()
    images2 = images2.cpu().detach().numpy()

    # calculate activations
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))

    act1 = torch.from_numpy(model.predict(images1)) ## The length of act1 is 2048. We assume this vector can be approximated by multivariate normal distribution
    act2 = torch.from_numpy(model.predict(images2))

    mu1, sigma1 = torch.mean(act1, dim=0), torch.cov(act1.T)
    mu2, sigma2 = torch.mean(act2, dim=0), torch.cov(act2.T)
    # calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean ,_ = sqrtm(torch.matmul(sigma1, sigma2) ,disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = sqrtm(torch.matmul((sigma1 + offset) ,(sigma2 + offset)))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid






