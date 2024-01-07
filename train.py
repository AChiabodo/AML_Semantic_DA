#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from GTA5 import GTA5
from utils import ExtCompose, ExtResize, ExtToTensor, ExtTransforms, ExtRandomHorizontalFlip ,ExtColorJitter, ExtScale, ExtRandomCrop, ExtGaussianBlur, Args
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode


logger = logging.getLogger()


def val(args, model, dataloader, writer = None , epoch = None, step = None):
    
    """
    Evaluate the current model on the validation set by computing:
    - Precision per pixel
    - Mean Intersection over Union (mIoU)
    - mIoU per class

    Args:
    - ...
    - epoch: Current training epoch.
    - step: Current training step.
    """
    
    print('start val!')
    
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        random_sample = random.randint(0, len(dataloader) - 1)
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            
            if i == random_sample and writer is not None:
                colorized_predictions , colorized_labels = CityScapes.visualize_prediction(predict, label)
                writer.add_image('eval%d/iter%d/predicted_eval_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/correct_eval_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/eval_original _data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')

            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
            
        #compute metrics    
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = [] #loss for each batch
        image_number = random.randint(0, len(dataloader_train) - 1)
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

                if i == image_number and epoch % 2 == 0: #saves the first image in the batch to tensorboard
                    print('epoch {}, iter {}, loss1: {}, loss2: {}, loss3: {}'.format(epoch, i, loss1, loss2, loss3))
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(output, label)
                    writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')

            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent large gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Memory cleanup
            torch.cuda.empty_cache()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        """
        Evaluate the model and save it if performance has improved.
        """
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, writer, epoch, step)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    #final evaluation
    val(args, model, dataloader_val, writer, epoch, step)

def train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, target_dataloader_val, layer=0):
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

    # 1. Initialization
    writer = SummaryWriter(comment=args.optimizer)
    scaler = amp.GradScaler() # Automatic Mixed Precision
    d_lr = 1e-4 # Discriminator learning rate
    max_lam = 0.0025 # Maximum value for the lambda parameter (used to balance the two losses)

    # 2. Discriminator Setup
    discr = torch.nn.DataParallel(BiSeNetDiscriminator(num_classes=args.num_classes)).cuda() 
    discr_optim = torch.optim.Adam(discr.parameters(), lr=d_lr, betas=(0.9, 0.99))

    # 3. Loss Functions
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    discr_loss_func = torch.nn.MSELoss()
    #discr_loss_func = torch.nn.BCEWithLogitsLoss()
    
    # 4. Training Loop for each epoch
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):

        # 4.1. Adjust Learning Rates
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        discr_lr = poly_lr_scheduler(discr_optim, d_lr, iter=epoch, max_iter=args.num_epochs)

        # 4.2. Adjust Lambda
        #lam = max_lam * ((epoch) / args.num_epochs) ** 0.9
        #lam = max_lam if epoch > 10 else max_lam * ( 1 + ((epoch - 10) / (args.num_epochs)))
        lam = (max_lam) #* (1 + np.sin(np.pi / 2 * epoch / 50)) / 2

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

        # 4.6. Save the average loss for the epoch
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # 4.7. Save other parameters on tensorboard
        writer.add_scalar('epoch/loss_epoch_discr', float(np.mean(loss_discr_record)), epoch)
        writer.add_scalar('train/lambda', float(lam), epoch)
        writer.add_scalar('train/discr_lr', float(discr_lr), epoch)
        writer.add_scalar('train/g_lr', float(lr), epoch)

        # 4.8. Save a checkpoint of the model every {args.checkpoint_step} epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        # 4.9. Evaluate the model on the validation set every {args.validation_step} epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, target_dataloader_val, writer, epoch, step, max_miou)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

   #final evaluation
    val(args, model, target_dataloader_val, writer, epoch, step, max_miou)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
                       help='Select between simple training (train), training with Domain Adaptation (train_da) or testing an already trained model (test)'
    )

    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='STDCNet813',
                       help='Select the backbone to use for the model. Supported backbones: STDCNet813'                       
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='pretrained_weights\STDCNet813M_73.91.tar',
                      help='Path to the pretrained weights of the backbone'
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
                       help='Whether to use the last convolutional layer of the backbone'
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,#300
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=5,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to modelwork')
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to modelwork')
    parse.add_argument('--batch_size',
                       type=int,
                       default=2,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.001, #0.01
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default='trained_models',
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    parse.add_argument('--resume',
                       type=str2bool,
                       default=False,
                       help='Define if the model should be trained from scratch or from a trained model')
    parse.add_argument('--dataset',
                          type=str,
                          default='CROSS_DOMAIN',
                          help='CityScapes, GTA5 or CROSS_DOMAIN. Define on which dataset the model should be trained and evaluated.')
    parse.add_argument('--resume_model_path',
                       type=str,
                       default='',
                       help='Define the path to the model that should be loaded for training. If void, the last model will be loaded.')
    parse.add_argument('--data_transformations',
                       type=int,
                       default=0,
                       help='Select transformations to be applied on the dataset images (0: no transformations, 1 : data augmentation)'
    )
    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    # to be changed
    if args.dataset == 'GTA5':
        args.crop_height, args.crop_width = 526 , 957
    
    # 2. Data Transformations Selection

    trans = Args(data_transformations=0, crop_height=256, crop_width=256)

    if trans.data_transformations == 0:
        transformations = ExtCompose([ExtScale(0.5, interpolation=Image.BILINEAR), ExtToTensor()])
        target_transformations = ExtCompose([ExtScale(0.5, interpolation=Image.BILINEAR), ExtToTensor()])
    elif trans.data_transformations == 1:
        transformations = ExtCompose([
            ExtScale(random.choice([0.75, 1, 1.25, 1.5, 1.75, 2]), interpolation=Image.BILINEAR),
            ExtRandomCrop((args.crop_height, args.crop_width)),
            ExtRandomHorizontalFlip(),
            ExtToTensor()
        ])
        target_transformations = ExtCompose([ExtScale(0.5, interpolation=Image.BILINEAR), ExtToTensor()])
    elif trans.data_transformations == 2:
        transformations = ExtCompose([
            ExtScale(random.choice([0.75, 1, 1.25, 1.5, 1.75, 2]), interpolation=Image.BILINEAR),
            ExtRandomCrop((args.crop_height, args.crop_width)),
            ExtRandomHorizontalFlip(),
            ExtGaussianBlur(p=0.5, radius=1),
            ExtColorJitter(p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
            ExtToTensor()
        ])
        target_transformations = ExtCompose([ExtScale(0.5, interpolation=Image.BILINEAR), ExtToTensor()])

    eval_transformations = ExtCompose([ExtScale(0.5, interpolation=Image.BILINEAR), ExtToTensor()])
     
    if args.dataset == 'CityScapes':
        print('training on CityScapes')
        train_dataset = CityScapes(split = 'train',transforms=transformations)
        val_dataset = CityScapes(split='val',transforms=eval_transformations)

    elif args.dataset == 'GTA5':
        print('training on GTA5')
        train_dataset = GTA5(root='dataset',split="train",transforms=transformations,labels_source="cityscapes")
        val_dataset = GTA5(root='dataset',split="eval",transforms=eval_transformations,labels_source="cityscapes")

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on CityScapes and validating on GTA5')
        train_dataset = GTA5(root='dataset',transforms=transformations,labels_source="cityscapes")
        val_dataset = CityScapes(split = 'val',transforms=transformations)
    else:
        print('not supported dataset \n')
        return None
    
    dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)
    dataloader_val = DataLoader(val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    if args.resume == 'False':
        try:
            if args.resume_model_path == '':
                args.resume_model_path = os.path.join(args.save_model_path, 'best.pth')
            model.load_state_dict(torch.load(args.resume_model_path))
            print('successfully resume model from %s' % args.resume_model_path)
        except Exception as e:
            print(e)
            print('resume failed, try again')
            return None

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    if args.mode == 'train':
    # train code
        ## train loop
        train(args, model, optimizer, dataloader_train, dataloader_val)
    elif args.mode == 'train_da':
        train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val)
    elif args.mode == 'test':
            # 11.3. Evaluation of an already trained model on the Validation Set
            writer = SummaryWriter(comment=args.comment)
            val(args, model, dataloader_val,writer=writer,epoch=0,step=0)
    else :
            print('not supported mode \n')
            return None
if __name__ == "__main__":
    main()