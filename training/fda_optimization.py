##########################################################################################
#                       SIMPLE SCHEMA
# import numpy as np
#import cv2
#from scipy.fftpack import fftn, ifftn, fftshift, ifftshift

#def FDA(source_img, target_img):
    # Step 1: Apply FFT to source and target images
#  source_fft = fftn(source_img)
#    target_fft = fftn(target_img)

    # Step 2: Replace the low frequency part of the source amplitude with that from the target
# #  source_amp = np.abs(source_fft)
#   target_amp = np.abs(target_fft)
#   source_phase = np.angle(source_fft)
#   target_phase = np.angle(target_fft)

    # Compute the amplitude of the source and target images
#   source_amp = fftshift(source_amp)
#    target_amp = fftshift(target_amp)

    # Define the radius for the low frequency region
#    h, w, _ = source_img.shape
#    cy, cx = int(h/2), int(w/2)
#    radius = min(cy, cx)

    # Create a circular mask for the low frequency region
#    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
#   mask = x*x + y*y <= radius*radius

    # Replace the low frequency part of the source amplitude with that from the target
#    source_amp = source_amp * (1-mask) + target_amp * mask

    # Step 3: Apply inverse FFT to the modified source spectrum
#    source_fft_amp_phase = source_amp * np.exp(1j*source_phase)
#    target_fft_amp_phase = target_amp * np.exp(1j*target_phase)

    # Apply the inverse FFT
#    source_img_ifft = np.real(ifftn(ifftshift(source_fft_amp_phase)))
#   target_img_ifft = np.real(ifftn(ifftshift(target_fft_amp_phase)))
#
#  return source_img_ifft, target_img_ifft
# 
# 
##########################################################################################################

# LIBRARIES
import torch
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
import random
import os
from tensorboardX import SummaryWriter

# PERSONAL
# Datasets
from datasets.cityscapes import CityScapes
# Utils
from utils.general import poly_lr_scheduler, load_ckpt, preprocess_image
from eval import evaluate_and_save_model

import fda


def train(args, model, optimizer, dataloader_train, dataloader_val, comment=''):
    """
    Train the model on the selected training set and evaluate it on the validation set.
    In the end, save the best model according to the mIoU on the validation set.

    Depending on the previously setted dataloader, this function can be used for:
    - Option 1: Simple Training, to get a baseline
    - Option 2: Training with Data Augmentation, to improve the performance of the model
    """

    #Initialization
    writer = SummaryWriter(comment=comment) # Tensorboard writer
    scaler = amp.GradScaler() # Automatic Mixed Precision
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0 # Best mIoU on the validation set
    step = 0 # Number of iterations

    # 1. Resume Model from Checkpoint
    if args.resume:
        try:
            if args.resume_model_path == '':
                args.resume_model_path = os.path.join(args.save_model_path, 'best.pth')
                print('No model path specified. Loading the best model trained so far: {}'.format(args.resume_model_path))
            max_miou, starting_epoch = load_ckpt(args, optimizer=optimizer, model=model)
            print('successfully resume model from %s' % args.resume_model_path)
        except Exception as e:
            print(e)
            print('resume failed, try again')
            return None

    # 2. Training Loop 
    for epoch in range(starting_epoch,args.num_epochs):

        # 2.1. Adjust Learning Rate
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        # 2.2. Set up the model to train mode
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = [] # Loss for each batch

        # 2.3. Select a random image to save to tensorboard
        image_number = random.randint(0, len(dataloader_train) - 1) 

        # 2.4. Training Loop for each batch
        for i, (data, label) in enumerate(dataloader_train):

            # 2.4.1. Load data and label to GPU
            data = data.cuda()
            label = label.long().cuda()
            
            #2.4.X  Apply FDA
            
            # Scegli un'immagine target casuale dal dataset Cityscapes
            target_im = random.choice(CityScapes).cuda() 
            
            for j in range(data.size(0)):  # per ogni immagine nel batch
                # Preprocessa l'immagine
                data[j] = preprocess_image(data[j])
                # applica fda
                data[j] = fda.fda(data[j], target_im, beta=0.005) 

            # 2.4.2. Zero the gradients
            optimizer.zero_grad()

            with amp.autocast():
                # 2.4.2.1. Forward pass -> multi-scale outputs
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

                # 2.4.2.2. Save the randomly selected image in the batch to tensorboard
                if i == image_number and epoch % 2 == 0: #saves the first image in the batch to tensorboard
                    print('epoch {}, iter {}, loss1: {}, loss2: {}, loss3: {}'.format(epoch, i, loss1, loss2, loss3))
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(output, label)
                    colorized_predictions_16 , _ = CityScapes.visualize_prediction(out16, label)
                    colorized_predictions_32 , _ = CityScapes.visualize_prediction(out32, label)

                    writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')
                    writer.add_image('epoch%d/iter%d/predicted_labels_16' % (epoch, i), np.array(colorized_predictions_16), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/predicted_labels_32' % (epoch, i), np.array(colorized_predictions_32), step, dataformats='HWC')

            # 2.4.3. Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 2.4.4. Update the progress bar
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            # 2.4.5. Save the loss for the batch
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()

        # 2.5. Save the average loss for the epoch
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # 2.6. Save a checkpoint of the model every {args.checkpoint_step} epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        # 2.7. Evaluate the model on the validation set every {args.validation_step} epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            max_miou = evaluate_and_save_model(args, model, dataloader_val, writer, epoch, step, max_miou)
    
    # 3. Final Evaluation
    max_miou = evaluate_and_save_model(args, model, dataloader_val, writer, epoch, step, max_miou)
