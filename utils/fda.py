"""
  # fda.py #

  This module provides the implementation of the Fourier Domain Adaptation (FDA) method,
  to adapt the appearance of source images to the target domain.
  
  The key idea of FDA is to swap the low-frequency components of the amplitude spectrum 
  of the source images with those of the target images, while preserving the phase spectrum.
  => semantic information is preserved
  => appearance is adapted to the target domain
"""
import torch
from torch import tensor
from PIL import Image
import numpy as np
import torch.nn as nn

def FDA_source_to_target( src_img: tensor, trg_img: tensor, beta=0.1 ):
    """
    Pperforms Fourier Domain Adaptation (FDA) from source to target domain.

    This function takes in source and target images and swaps the low-frequency components
    of the amplitude spectrum of the source images with those of the target images, while
    preserving the phase spectrum of the source images.

    Args:
    - src_img: source image (tensor)
    - trg_img: target image (tensor)
    - beta: hyperparameter to control the amount of adaptation (float)
            the size of the low-frequency window to be swapped
    """

    # 0. Resize the target image to match the size of the source image
    src_clone = src_img.clone()
    trg_clone = trg_img.clone()
    trg_clone = trg_clone.resize( (src_clone.size(3), src_clone.size(2)), Image.BICUBIC )

    # 1. Compute the 2D Fourier Transform of the source and target images
    fft_src = torch.rfft( src_clone, signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_clone, signal_ndim=2, onesided=False )

    # 2. Extract the amplitude and phase components of the Fourier Transform
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # 3. Swap the low-frequency components of the amplitude spectrum of the source images with those of the target images
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), beta )

    # 4. Recompose the Fourier Transform of the source image with the new amplitude and the original phase
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # 5. Compute the inverse Fourier Transform to obtain the adapted source image
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

def extract_ampl_phase(fft_im):
    """
      Extract amplitude and phase from the Fourier Transform of an image.
    """
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, beta=0.1 ):
    """
      Swap the low-frequency components of the amplitude spectrum of the source images
      with those of the target images.
    """
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*beta)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src


#TODO: CHECK THE FOLLOWING FUNCTIONS

# Preprocessing an image
def preprocess_image(image):
    """
    Esegue il preprocessing su un'immagine.
    """
    # Sottrai la media
    mean = np.mean(image)
    image = image - mean

    # Normalizza l'immagine
    std = np.std(image)
    image = image / std

    return image

#function for loss entropy fro cityscapes (FDA)
class EntropyMinimizationLoss(nn.Module):
    def __init__(self, h):
        super(EntropyMinimizationLoss, self).__init__()
        self.h = h

    def forward(self, phi_w, x_t):
        """
        Computes entropy minimization loss.

        Args:
            phi_w (torch.Tensor): Parameter tensor.
            x_t (torch.Tensor): Target data tensor.

        Returns:
            torch.Tensor: Entropy minimization loss.
        """
        log_phi_w = torch.log(phi_w)
        term1 = -self.h * phi_w
        term2 = log_phi_w
        rho_term = torch.sum(torch.abs(term1 - term2))

        return rho_term

# Example usage:
h_value = 0.5  # Adjust as needed
loss_fn = EntropyMinimizationLoss(h_value)

# Assuming phi_w and x_t are your tensors
phi_w = torch.randn(10, requires_grad=True)
x_t = torch.randn(10)

entropy_loss = loss_fn(phi_w, x_t)
#print(f"Entropy Minimization Loss: {entropy_loss.item()}")