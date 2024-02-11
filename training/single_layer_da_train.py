# LIBRARIES
import torch
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
import random
import os
from tensorboardX import SummaryWriter

# PERSONAL
# Models
from model.model_stages import BiSeNetDiscriminator
# Datasets
from datasets.cityscapes import CityScapes
# Utils
from utils import poly_lr_scheduler, save_ckpt, load_ckpt
from eval import evaluate_and_save_model



def train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, target_dataloader_val, comment='', layer=0,starting_epoch=0):
    """
    Train the model using `Domain Adaptation (DA)` for semantic segmentation tasks. 
    
    The function adapts a segmentation model from a source domain (with labeled data)
    to a target domain (with unlabeled data), aiming to improve performance on the last. 
    This is achieved through `adversarial training` involving:
    1. The `generator` (the segmentation model): trained to perform well on the source domain 
       while also trying to fool the discriminator into believing that its predictions on the 
       target domain are from the source domain.
    2. The `discriminator` (a binary classifier): trained to distinguish between the 
       generator's predictions on the source and target domains.

    The function iteratively updates both the generator and the discriminator.
    Each epoch involves:
    1. Generator Training:
        - Update the generator to minimize the segmentation loss on the source data.
        - Update the generator to fool the discriminator into believing that its predictions on the target data are from the source domain.
    2. Discriminator Training:
        - Update the discriminator to distinguish between the generator's predictions on the source and target domains.
    
    Args:
    - ...
    - layer: Indicates which layer's output to use for domain adaptation in the discriminator.
    """

    # 1.1 Parameters Initialization
    writer = SummaryWriter(comment=comment) # Tensorboard writer
    scaler = amp.GradScaler() # Automatic Mixed Precision
    d_lr = 2e-4 # Discriminator learning rate
    max_lam = 0.0015 # Maximum value for the lambda parameter (used to balance the two losses)
    max_miou = 0
    step = 0

    # 1.2 Discriminator Initialization
    discr = torch.nn.DataParallel(BiSeNetDiscriminator(num_classes=args.num_classes)).cuda() 
    discr_optim = torch.optim.Adam(discr.parameters(), lr=d_lr, betas=(0.9, 0.99))

    # 2. Resume Model from Checkpoint
    if args.resume:
        try:
            max_miou, starting_epoch = load_ckpt(args, optimizer=optimizer, model=model, discriminator=discr, discriminator_optimizer=discr_optim)
            print('successfully resume model from %s' % args.resume_model_path)
        except Exception as e:
            print(e)
            print('resume failed, try again')
            return None

    # 3. Loss Functions
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    #discr_loss_func = torch.nn.MSELoss()
    discr_loss_func = torch.nn.BCEWithLogitsLoss()
    
    # 4. Training Loop for each epoch
    for epoch in range(starting_epoch,args.num_epochs):

        # 4.1. Adjust Learning Rates
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        discr_lr = poly_lr_scheduler(discr_optim, d_lr, iter=epoch, max_iter=args.num_epochs)

        # 4.2. Adjust Lambda
        lam = (max_lam)

        # 4.3. Set up the model to train mode
        model.train()
        tq = tqdm(total=min(len(source_dataloader_train),len(target_dataloader_train)) * args.batch_size)
        tq.set_description('epoch %d,G-lr %f, D-lr %f, lam %f' % (epoch, lr, discr_lr, lam))
        # Losses for each batch
        loss_record = []
        loss_discr_record = []
        # Utility functions
        softmax_func = torch.nn.functional.softmax
        dsource_labels, dtarget_labels = torch.zeros, torch.ones

        # 4.4. Select a random image to save to tensorboard
        image_number = random.randint(0, min(len(source_dataloader_train),len(target_dataloader_train)) - 1)

        # 4.5. Training Loop for each batch
        for i, (source_data, target_data) in enumerate(zip(source_dataloader_train,target_dataloader_train)):

            # 4.5.1. Load data and label to GPU
            source_data,source_label = source_data
            target_data,_ = target_data
            source_data = source_data.cuda()
            source_label = source_label.long().cuda()
            target_data = target_data.cuda()

            # 4.5.2. Zero the gradients
            optimizer.zero_grad()
            discr_optim.zero_grad()
            
            ###################
            # Train Generator #
            ###################

            # TG.1. Set up the discriminator to eval mode
            discr.module.train_params(False)

            # TG.2. Train to minimize the segmentation loss on the source data
            with amp.autocast():
                
                # TG.2.1. Forward pass -> multi-scale outputs
                s_output, s_out16, s_out32 = model(source_data)
                
                # TG.2.2. Compute the segmentation loss
                loss1 = loss_func(s_output, source_label.squeeze(1))
                loss2 = loss_func(s_out16, source_label.squeeze(1))
                loss3 = loss_func(s_out32, source_label.squeeze(1))
                loss = loss1 + loss2 + loss3
            
            # TG.3. Backward pass of the segmentation loss
            scaler.scale(loss).backward()
            
            # TG.4. Train to fool the discriminator using the target training data
            with amp.autocast():
                
                # TG.4.1. Forward pass -> multi-scale outputs
                t_output, t_out16, t_out32 = model(target_data)

                # TG.4.2. Select the right output for the discriminator
                # (depending on the layer selected for domain adaptation)
                # and get the discriminator's prediction
                if layer == 0:
                    dom = discr(softmax_func(t_output, dim=1))
                elif layer == 1:
                    dom = discr(softmax_func(t_out16, dim=1))
                elif layer == 2:
                    dom = discr(softmax_func(t_out32, dim=1))
                else:
                    raise ValueError('layer should be 0, 1 or 2')

                # TG.4.3. Compute the discriminator loss for the selected layer:
                # a perfect discriminator would predict all 1s for images' outputs coming from the target domain;
                # however, we're training G to fool D into believing that they're actually coming from the source domain
                # (i.e. we would like D to predict all 0s)
                td_loss_fooled = discr_loss_func(dom, dsource_labels(dom.shape,dtype=torch.float).cuda())

                # TG.4.4. Scale the discriminator loss by lambda:
                # to limit the impact of the discriminator on the generator
                # whose primary goal is still to minimize the segmentation loss
                d_loss = td_loss_fooled * lam

            # TG.5. Backward pass of the discriminator loss
            scaler.scale(d_loss).backward()

            #######################
            # Train Discriminator #
            #######################

            # TD.1. Set up the discriminator to train mode
            discr.module.train_params(True)

            # TD.2. Select the right output for the discriminator
            # (based on the layer selected for domain adaptation)
            if layer == 0:
                s_output = s_output.detach()
                t_output = t_output.detach()
            elif layer == 1:
                s_output = s_out16.detach()
                t_output = t_out16.detach()
            elif layer == 2:
                s_output = s_out32.detach()
                t_output = t_out32.detach()
            else:
                raise ValueError('layer should be 0, 1 or 2')
            
            # TD.3. Train to distinguish between the source and target domains:
            # A perfect discriminator would predict all 0s for images' outputs coming from the source domain
            # and all 1s for images' outputs coming from the target domain.
            # (Losses are divided by 2 to avoid the discriminator overpowering the generator)
            
            with amp.autocast():
                # TD.3.1. Train on the source domain
                dom = discr(softmax_func(s_output, dim=1))
                sd_loss = discr_loss_func(dom, dsource_labels(dom.shape,dtype=torch.float).cuda()) / 2
            scaler.scale(sd_loss).backward()  # Backward pass of the source domain loss
            
            with amp.autocast():
                # TD.3.2. Train on the target domain
                dom = discr(softmax_func(t_output , dim=1))
                td_loss = discr_loss_func(dom, dtarget_labels(dom.shape,dtype=torch.float).cuda()) / 2
            scaler.scale(td_loss).backward() # Backward pass of the target domain loss

            # 4.5.3. Save the randomly selected image in the batch to tensorboard
            if i == image_number and epoch % 2 == 0: #saves the first image in the batch to tensorboard
                print('epoch {}, iter {}, loss1: {}, loss2: {}, loss3: {}, d_loss_fool: {}, d_loss: {}'.format(epoch, i, loss1, loss2, loss3, d_loss, sd_loss+td_loss))
                colorized_predictions , colorized_labels = CityScapes.visualize_prediction(s_output, source_label)
                colorized_predictions_16 , _ = CityScapes.visualize_prediction(s_out16, source_label)
                colorized_predictions_32 , _ = CityScapes.visualize_prediction(s_out32, source_label)

                writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(source_data[0].cpu(),dtype='uint8'), step, dataformats='CHW')
                writer.add_image('epoch%d/iter%d/predicted_labels_16' % (epoch, i), np.array(colorized_predictions_16), step, dataformats='HWC')
                writer.add_image('epoch%d/iter%d/predicted_labels_32' % (epoch, i), np.array(colorized_predictions_32), step, dataformats='HWC')

            # 4.5.4. Update the generator and the discriminator
            scaler.step(optimizer)
            scaler.step(discr_optim)
            scaler.update()

            # 4.5.5. Compute the total loss for the batch
            tot_g_loss = loss + d_loss # Generator's total loss
            tot_d_loss = sd_loss + td_loss # Discriminator's total loss

            # 4.5.6. Update the progress bar
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % tot_g_loss)
            step += 1
            
            # 4.5.7. Save the loss for the batch
            loss_record.append(tot_g_loss.item())
            loss_discr_record.append(tot_d_loss.item())
        tq.close()

        # 4.6. Save a checkpoint of the model every {args.checkpoint_step} epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            #torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))
            save_ckpt(args=args,model=model, optimizer=optimizer,cur_epoch=epoch,best_score= max_miou,name='latest.pth',discriminator_optimizer=discr_optim,discriminator=discr)
        
        # 4.7. Save the average loss for the epoch
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # 4.8. Save other parameters on tensorboard
        writer.add_scalar('epoch/loss_epoch_discr', float(np.mean(loss_discr_record)), epoch)
        writer.add_scalar('train/lambda', float(lam), epoch)
        writer.add_scalar('train/discr_lr', float(discr_lr), epoch)
        writer.add_scalar('train/g_lr', float(lr), epoch)

        # 4.9. Evaluate the model on the validation set every {args.validation_step} epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            max_miou = evaluate_and_save_model(args, model, target_dataloader_val, writer, epoch, step, max_miou)
    
    # 5. Final Evaluation
    max_miou = evaluate_and_save_model(args, model, target_dataloader_val, writer, epoch, step, max_miou)
