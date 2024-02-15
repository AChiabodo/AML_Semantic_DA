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
import os
from torch.utils.tensorboard import SummaryWriter
from model.model_stages import BiSeNet
from utils.general import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, save_ckpt
from datasets.cityscapes import CityScapes

#################
# FDA TRANSFORM #
#################

def FDA_source_to_target( src_img: tensor, trg_img: tensor, beta=0.05 ):
    """
      Performs Fourier Domain Adaptation (FDA) from source to target domain.

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

#################
# LOSS FUNCTION #
#################

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
    
##################
# MBT ADAPTATION #
##################
    
def test_mbt(args, dataloader_val, comment,
              path_b1 = 'trained_models\\test_norm_fda_0.01\\best.pth',
              path_b2 = 'trained_models\\test_norm_fda_0.05\\best.pth',
              path_b3 = 'trained_models\\test_norm_fda_0.09\\best.pth',
              save_pseudo_labels = True,
              save_path = 'dataset\\Cityscapes\\pseudo_labels'
            ):
    """
      Evaluation of the Segmentation Networks Adapted with Multi-band Transfer (multiple betas)

      MBT is a technique that leverages Fourier Domain Adaptation (FDA) to reduce the domain gap 
      between synthetic and real images, and uses the mean prediction of multiple models trained 
      with different spectral domain sizes to generate pseudo-labels for the target domain.

      The pseudo-labels will be used to train the model in a self-learning fashion.
      => To avoid overfitting, we want to filter out the low-confidence predictions from the
      pseudo labels, by only accepting those that, for each semantic class, are either:
      - in the top 66%
      - or above 0.9
      Any labels with a probability below the threshold are set to 255 (ignored class). 

      Args:
      - ...
      - path_b1: path to the model trained with beta=0.01 (str)
      - path_b2: path to the model trained with beta=0.05 (str)
      - path_b3: path to the model trained with beta=0.09 (str)
      - save_pseudo_labels: flag to save the pseudo-labels (bool)
      - save_path: path to save the pseudo-labels (str)
    """

    with torch.no_grad(): # No need to track the gradients during validation

      # 0. Initialization
      backbone = args.backbone
      n_classes = args.num_classes
      pretrain_path = args.pretrain_path
      use_conv_last = args.use_conv_last

      precision_record = [] # list to store precision per pixel
      hist = np.zeros((n_classes, n_classes)) # confusion matrix (for mIoU)

      # 1. Pseudo-labels Initialization
      if save_pseudo_labels:
          
          # 1.1. Create the folder to save the pseudo-labels
          if not os.path.exists(save_path):
              os.makedirs(save_path)

          # 1.2. Initialize the arrays to store the pseudo-labels
          predicted_labels = np.zeros((len(dataloader_val), 512, 1024), dtype=np.uint8)
          predicted_probs = np.zeros((len(dataloader_val), 512, 1024), dtype=np.float32)
          image_names = []

          print('Pseudo-labels will be saved in: ', save_path)

      # 2. Load the models trained with different betas
      checkpoint_b1 = torch.load(path_b1)      
      model_b1 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
      model_b1.load_state_dict(checkpoint_b1['model_state_dict'])
      model_b1.cuda()
      model_b1.eval()

      checkpoint_b2 = torch.load(path_b2)
      model_b2 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
      model_b2.load_state_dict(checkpoint_b2['model_state_dict'])
      model_b2.cuda()
      model_b2.eval()

      checkpoint_b3 = torch.load(path_b3)
      model_b3 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
      model_b3.load_state_dict(checkpoint_b3['model_state_dict'])
      model_b3.cuda()
      model_b3.eval()

      print('Testing MBT Adaptation...')

      # 3. Iterate over the validation dataset
      for i, (data, label) in enumerate(dataloader_val):

          # 3.1. Load data and label to GPU
          label = label.type(torch.LongTensor)
          data = data.cuda()
          label = label.long().cuda()

          # 3.2. Forward pass -> original scale output
          predict_b1, _, _ = model_b1(data)
          predict_b2, _, _ = model_b2(data)
          predict_b3, _, _ = model_b3(data)

          # 3.3. Compute the mean prediction
          predict = (predict_b1 + predict_b2 + predict_b3) / 3

          #################
          # PSEUDO-LABELS #
          #################
          if save_pseudo_labels:
              
              # For each image in the batch
              for j in range(predict.size(0)):
                  
                  # PL.1. Get the predicted label
                  pred = predict[j].squeeze(0) # Squash batch dimension
                  pred = reverse_one_hot(pred) # Convert to 2D tensor where each pixel is the class index
                  pred = np.array(pred.cpu()) # Convert to numpy array
                  predicted_labels[i*args.batch_size + j] = pred

                  # PL.2. Get the predicted probabilities
                  probs = np.max(predict[j].cpu().numpy(), axis=0)
                  predicted_probs[i*args.batch_size + j] = probs

                  # PL.3. Get the image name
                  path = dataloader_val.dataset.images[i*args.batch_size + j]
                  folder = path.split('\\')[-2]
                  name = path.split('\\')[-1]
                  image_names.append(folder + '\\' + name)

          # 3.4. Get the predicted label
          predict = predict.squeeze(0) # Squash batch dimension
          predict = reverse_one_hot(predict) # Convert to 2D tensor where each pixel is the class index
          predict = np.array(predict.cpu()) # Convert to numpy array

          # 3.5. Get the ground truth label
          label = label.squeeze()
          label = np.array(label.cpu())

          # 3.6. Compute precision per pixel and update the confusion matrix
          precision = compute_global_accuracy(predict, label)
          hist += fast_hist(label.flatten(), predict.flatten(), n_classes)
          precision_record.append(precision)

      # 4. Compute metrics
      precision = np.mean(precision_record)
      miou_list = per_class_iu(hist)
      miou = np.mean(miou_list)
      print('precision per pixel for test: %.3f' % precision)
      print('mIoU for test: %.3f' % miou)
      print('miou_list: ', miou_list)

      #################
      # PSEUDO-LABELS #
      #################
      if save_pseudo_labels:
      
          # PL.4. Filter out the low-confidence predictions
          # For each semantic class
          for i in range(n_classes):
              
              # PL.4.1. Get the probabilities for the current class
              probs = predicted_probs[:,:,i]
              
              # PL.4.2. Compute the threshold
              # - top 66% or above 0.9
              threshold = np.percentile(probs, 66, axis=0)
              threshold = np.where(threshold < 0.9, 0.9, threshold)
              
              # PL.4.3. Filter out the low-confidence predictions
              predicted_labels[probs < threshold] = 255

          # PL.5. Save the pseudo-labels
          for i, name in enumerate(image_names):
              
              # PL.5.1. Get the final path
              path = os.path.join(save_path, name)

              # PL.5.2. Save the pseudo-labeL as a H=1024 W=2048 RGB image
              im_colored = CityScapes.decode_target(predicted_labels[i])
              im_colored = Image.fromarray(im_colored)
              im_transformed = im_colored.resize((2048, 1024), Image.NEAREST)
              im_transformed.save(path)

          print('Pseudo-labels saved!')


      return precision, miou
      
