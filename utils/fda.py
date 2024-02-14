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
import torch.nn.functional as F

def FDA_source_to_target( src_img: tensor, trg_img: tensor, beta=0.05 ):
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

    # 0. Clone the source and target images
    src_clone = src_img.clone()
    trg_clone = trg_img.clone()

    # 1. Compute the 2D Fourier Transform of the source and target images
    # output: [batch, channel, height, width] where each value is a complex number
    fft_src = torch.view_as_real(torch.fft.fft2(src_clone))
    fft_trg = torch.view_as_real(torch.fft.fft2(trg_clone))

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
    src_in_trg = torch.fft.ifft2( torch.view_as_complex(fft_src_) )

    # 6. ATTENTION! Clamp the pixel values to be in the range [0, 255] 
    # (the Fourier Transform may introduce out-of-range values for very bright/dark areas)
    # This is done to avoid areas with random "burned" colors
    src_in_trg = torch.real(src_in_trg)
    src_in_trg = torch.clamp(src_in_trg, 0, 255)
    
    return src_in_trg.cuda()

def extract_ampl_phase(fft_im):
    """
      Extract amplitude and phase from the Fourier Transform of an image.
    
      The fft_im tensor has 5 dimensions: [batch, channel, height, width, complex]
      The last dimension is a complex number
      - fft_im[:,:,:,:,0] is the real part
      - fft_im[:,:,:,:,1] is the imaginary part
    """

    # 1. Compute the amplitude and phase components of the Fourier Transform
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha


def low_freq_mutate( amp_src, amp_trg, beta=0.1 ):
    """
      Swap the low-frequency components of the amplitude spectrum of the source images
      with those of the target images.
    """
    # 1. Get the size of the low-frequency window to be swapped
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*beta)  ).astype(int)     # get b

    # 2. Swap the low-frequency components of the amplitude spectrum
    amp_src[:,:,0:b,0:b] = amp_trg[:,:,0:b,0:b]

    # ATTENTION: We've intentionally omitted this part from the original implementation
    # It's not clear why also the high-frequency components are swapped
    # The results are better without this part
    # amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    # amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    # amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right

    return amp_src

#TODO: CHECK THE FOLLOWING FUNCTION

class EntropyMinimizationLoss(nn.Module):
    def __init__(self):
        super(EntropyMinimizationLoss, self).__init__()

    def forward(self, x, ita):
        """
        Computes entropy minimization loss.

        Args:
        - x: input tensor (tensor)
        - ita: hyperparameter to control the amount of entropy minimization (float)

        Returns:
            torch.Tensor: Entropy minimization loss.
        """

        P = F.softmax(x, dim=1)        # [B, 19, H, W]
        logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
        PlogP = P * logP               # [B, 19, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
        ent = ent / 2.9444         # change when classes is not 19
        # compute robust entropy
        ent = ent ** 2.0 + 1e-8
        ent = ent ** ita
        ent_loss_value = ent.mean()

        return ent_loss_value