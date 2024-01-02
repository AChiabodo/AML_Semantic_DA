#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet, BiSeNetDiscriminator
from cityscapes import CityScapes
from GTA5 import GTA5
from data_augmentation import ExtCompose, ExtResize, ExtToTensor, ExtTransforms, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from utils import str2bool
from tqdm import tqdm
import random
import os
from PIL import Image

logger = logging.getLogger()

def val(args, model, dataloader, writer = None , epoch = None, step = None):
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
            
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, target_dataloader_val, comment='', layer=0):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()
    d_lr = 1e-4
    max_lam = 0.0025
    discr = torch.nn.DataParallel(BiSeNetDiscriminator(num_classes=args.num_classes)).cuda()
    discr_optim = torch.optim.Adam(discr.parameters(), lr=d_lr, betas=(0.9, 0.99))

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    discr_loss_func = torch.nn.MSELoss()
    #discr_loss_func = torch.nn.BCEWithLogitsLoss()
    
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        discr_lr = poly_lr_scheduler(discr_optim, d_lr, iter=epoch, max_iter=args.num_epochs)
        #lam = max_lam * ((epoch) / args.num_epochs) ** 0.9
        #lam = max_lam if epoch > 10 else max_lam * ( 1 + ((epoch - 10) / (args.num_epochs)))
        lam = (max_lam) #* (1 + np.sin(np.pi / 2 * epoch / 50)) / 2
        model.train()
        tq = tqdm(total=min(len(source_dataloader_train),len(target_dataloader_train)) * args.batch_size)
        tq.set_description('epoch %d,G-lr %f, D-lr %f, lam %f' % (epoch, lr, discr_lr, lam))
        loss_record = []
        loss_discr_record = []
        image_number = random.randint(0, min(len(source_dataloader_train),len(target_dataloader_train)) - 1)

        softmax_func = torch.nn.functional.softmax
        dsource_labels , dtarget_labels = torch.ones , torch.zeros

        for i, (source_data, target_data) in enumerate(zip(source_dataloader_train,target_dataloader_train)):
            source_data,source_label = source_data
            target_data,_ = target_data
            source_data = source_data.cuda()
            source_label = source_label.long().cuda()
            target_data = target_data.cuda()
            optimizer.zero_grad()
            discr_optim.zero_grad()
            
            ##############################################
            # Train Generator
            ##############################################            
            discr.module.train_params(False)

            with amp.autocast():
                #train source data
                s_output, s_out16, s_out32 = model(source_data)
                
                ### Train Generator with source data
                loss1 = loss_func(s_output, source_label.squeeze(1))
                loss2 = loss_func(s_out16, source_label.squeeze(1))
                loss3 = loss_func(s_out32, source_label.squeeze(1))
                
                loss = loss1 + loss2 + loss3
            scaler.scale(loss).backward()
            
            with amp.autocast():
                ### Fool the discriminator by using source labels on target data
                t_output, t_out16, t_out32 = model(target_data)
                if layer == 0:
                    dom = discr(softmax_func(t_output, dim=1))
                elif layer == 1:
                    dom = discr(softmax_func(t_out16, dim=1))
                elif layer == 2:
                    dom = discr(softmax_func(t_out32, dim=1))
                else:
                    raise ValueError('layer should be 0, 1 or 2')
                
                td_loss_fooled = discr_loss_func(dom, dsource_labels(dom.shape,dtype=torch.float).cuda())
                
                d_loss = td_loss_fooled * lam
            scaler.scale(d_loss).backward()

            ##############################################
            # Train Discriminator
            ##############################################
            discr.module.train_params(True)

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
            with amp.autocast():
                dom = discr(softmax_func(s_output, dim=1))
                sd_loss = discr_loss_func(dom, dsource_labels(dom.shape,dtype=torch.float).cuda()) / 2
                
            scaler.scale(sd_loss).backward()    

            with amp.autocast():
                dom = discr(softmax_func(t_output , dim=1))
                td_loss = discr_loss_func(dom, dtarget_labels(dom.shape,dtype=torch.float).cuda()) / 2
                
            scaler.scale(td_loss).backward()

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

            scaler.step(optimizer)
            scaler.step(discr_optim)
            scaler.update()


            loss = loss + d_loss
            d_loss = sd_loss + td_loss
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            #writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            loss_discr_record.append(d_loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        writer.add_scalar('epoch/loss_epoch_discr', float(np.mean(loss_discr_record)), epoch)
        writer.add_scalar('train/lambda', float(lam), epoch)
        writer.add_scalar('train/discr_lr', float(discr_lr), epoch)
        writer.add_scalar('train/g_lr', float(lr), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, target_dataloader_val, writer, epoch, step)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    #final evaluation
    precision, miou = val(args, model, target_dataloader_val, writer, epoch, step)
    if miou > max_miou:
        max_miou = miou
        os.makedirs(args.save_model_path, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou val', miou, epoch)


def train(args, model, optimizer, dataloader_train, dataloader_val, comment=''):
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
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
                    colorized_predictions_16 , _ = CityScapes.visualize_prediction(out16, label)
                    colorized_predictions_32 , _ = CityScapes.visualize_prediction(out32, label)

                    writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')
                    writer.add_image('epoch%d/iter%d/predicted_labels_16' % (epoch, i), np.array(colorized_predictions_16), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/predicted_labels_32' % (epoch, i), np.array(colorized_predictions_32), step, dataformats='HWC')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, writer, epoch, step)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    #final evaluation
    precision, miou = val(args, model, dataloader_val, writer, epoch, step)
    if miou > max_miou:
        max_miou = miou
        os.makedirs(args.save_model_path, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou val', miou, epoch)


def parse_args():
    """Parse input arguments from command line"""
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train_da',
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
                       type=int, 
                       default=50,
                       help='Number of epochs to train the model'
    )
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number. Useful to resume training from a checkpoint'
    )
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often (epochs) to save a checkpoint of the model'
    )
    parse.add_argument('--validation_step',
                       type=int,
                       default=5,
                       help='How often (epochs) to evaluate the model on the validation set to check its performance'
    )
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to model'
    )
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to model'
    )
    parse.add_argument('--batch_size',
                       type=int,
                       default=4, #2
                       help='Number of images in each batch'
    )
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.001, #0.01
                        help='learning rate used during training to adjust the weights of the model'
    )
    parse.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='Number of threads used to load the data during training'
    )
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='Number of semantic classes to predict'
    )
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU id used for training'
    )
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training'
    )
    parse.add_argument('--save_model_path',
                       type=str,
                       default='trained_models',
                       help='path to save the trained model'
    )
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer to use for training (adam, sgd, rmsprop are supported)'
    )
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function to use for training (crossentropy, dice are supported)'
    )
    parse.add_argument('--resume',
                       type=str2bool,
                       default=False,
                       help='Define if the model should be trained from scratch or from a checkpoint'
    )
    parse.add_argument('--resume_model_path',
                       type=str,
                       default='',
                       help='Define the path to the model that should be loaded for training. If void, the best model trained so far will be loaded'
    )
    parse.add_argument('--dataset',
                          type=str,
                          default='CROSS_DOMAIN',
                          help='CityScapes, GTA5 or CROSS_DOMAIN. Define on which dataset the model should be trained and evaluated.'
    )
    parse.add_argument('--comment',
                       type=str,
                       default='test',
                       help='Comment to add to the log and on tensorboard to identify the model'
    )
    parse.add_argument('--data_transformations',
                       type=int,
                       default=0,
                       help='Select transformations to be applied on the dataset images (0: no transformations, 1 : data augmentation)'
    )
    return parse.parse_args()


# --dataset GTA5 --data_transformations 0 --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\test_print_features --resume False --comment test_print_features--mode train
# --mode train_da --dataset CROSS_DOMAIN --save_model_path trained_models\adv_single_layer_lam0.001_softmax_resumed --comment adv_single_layer_lam0.005_softmax --data_transformation 0 --batch_size 4 --learning_rate 0.002 --num_workers 4 --optimizer sgd --resume True --resume_model_path trained_models\avd_single_layer_lam0.005_softmax\best.pth
# & C:/Users/aless/Documents/Codice/AML_Semantic_DA/.venv/Scripts/python.exe c:/Users/aless/Documents/Codice/AML_Semantic_DA/train.py --mode train_da --dataset CROSS_DOMAIN --save_model_path trained_models\adv_single_layer_lam0.0025_MSELoss --comment adv_single_layer_lam0.0025_MSELoss --data_transformation 1 --batch_size 5 --learning_rate 0.0025 --num_workers 4 --optimizer sgd --resume True --resume_model_path trained_models\avd_single_layer_lam0.005_softmax\best.pth --crop_height 526 --crop_width 957
def main():

    # 1. Initialization
    args = parse_args()

    # 
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    # to be changed
    if args.dataset == 'GTA5':
        args.crop_height, args.crop_width = 526 , 957
    
    # 2. Data Transformations Selection
    match args.data_transformations:
        case 0:
            """
            No Data Augmentation
            - Images are resized to 0.5 of their original size to reduce computational cost
            - No extra transformations are applied to the dataset
            """
            transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()]) #ExtRandomHorizontalFlip(),
            target_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
        case 1:
            """
            Data Augmentation
            - Images are resized to 0.5 of their original size to reduce computational cost
            - Randomly flipped horizontally
            - Randomly cropped to the desired size
            """
            transformations = ExtCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomHorizontalFlip(),
                ExtRandomCrop((args.crop_height, args.crop_width)),
                ExtToTensor()])
            target_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
    
    """The Validation Set is also resized to 0.5 of its original size"""
    eval_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
    
    # 3. Datasets Selection
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        train_dataset = CityScapes(split = 'train',transforms=transformations)
        val_dataset = CityScapes(split='val',transforms=eval_transformations)

    elif args.dataset == 'GTA5':
        print('training on GTA5')
        train_dataset = GTA5(root='dataset',split="train",transforms=transformations,labels_source='cityscapes')
        val_dataset = GTA5(root='dataset',split="eval",transforms=eval_transformations,labels_source='cityscapes')

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on GTA and validating on Cityscapes')
        train_dataset = GTA5(root='dataset',transforms=transformations)
        target_dataset_train = CityScapes(split = 'train',transforms=target_transformations)
        val_dataset = CityScapes(split = 'val',transforms=eval_transformations)
    else:
        print('not supported dataset \n')
        return None
    
    # 4. Dataloaders Setup
    source_dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)
    if args.dataset == 'CROSS_DOMAIN':
        target_dataloader_train = DataLoader(target_dataset_train,
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
    
    # 5. Model Setup
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    # 6. Resume Model from Checkpoint
    if args.resume or args.mode == 'test':
        try:
            # 6.1. If no model path is specified
            if args.resume_model_path == '':
                # Load the best model trained so far
                args.resume_model_path = os.path.join(args.save_model_path, 'best.pth')
                print('No model path specified. Loading the best model trained so far: {}'.format(args.resume_model_path))
            # 6.2. Load the model
            model.load_state_dict(torch.load(args.resume_model_path))
            print('successfully resume model from %s' % args.resume_model_path)
        except Exception as e:
            print(e)
            print('resume failed, try again')
            return None

    # 7. GPU Parallelization
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # 8. Optimizer Selection
    if args.optimizer == 'rmsprop':
        """
        Root Mean Square Propagation
        - Adapts learning rates based on moving average of squared gradients
        - Good for noisy or non-stationary objectives
        """
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        """
        Stochastic Gradient Descent
        - Fixed learning rate, requires manual tuning
        - Momentum can be added to accelerate gradients in the right direction
        - Weight decay can be added to regularize the weights
        """
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        """
        Adaptive Moment Estimation
        - Computes adaptive learning rates for each parameter
        - Incorporates momentum to escape local minima
        - Well suited for problems with large datasets and parameters
        """
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        print('not supported optimizer \n')
        return None
    
    # 9. Comment for Tensorboard
    if args.comment == '':
        args.comment = "_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate)

    # 10. Start Training or Evaluation
    match args.mode:
        case 'train':
            # 10.1. Simple Training on Source Dataset
            train(args, model, optimizer, source_dataloader_train, dataloader_val, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case 'train_da':
            # 10.2. Training with Domain Adaptation
            train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment)
        case 'test':
            # 10.3. Evaluation of an already trained model on the Validation Set
            writer = SummaryWriter(comment=args.comment)
            val(args, model, dataloader_val,writer=writer,epoch=0,step=0)
        case _:
            print('not supported mode \n')
            return None

if __name__ == "__main__":
    main()