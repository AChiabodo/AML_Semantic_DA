#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet, BiSeNetDiscriminator
from cityscapes import CityScapes
from GTA5 import GTA5
from utils import ExtCompose, ExtResize, ExtToTensor, ExtTransforms, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop
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

def train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, target_dataloader_val, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()
    
    discr = BiSeNetDiscriminator(num_classes=args.num_classes).cuda()
    discr_optim = torch.optim.Adam(discr.parameters(), lr=0.0002, betas=(0.9, 0.99))

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    discr_loss_func = torch.nn.MSELoss()

    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        discr_lr = poly_lr_scheduler(discr_optim, 0.0002, iter=epoch, max_iter=args.num_epochs)

        model.train()
        tq = tqdm(total=min(len(source_dataloader_train),len(target_dataloader_train)) * args.batch_size)
        tq.set_description('epoch %d,G lr %f, D lr %f' % (epoch, lr, discr_lr))
        loss_record = []
        image_number = random.randint(0, min(len(source_dataloader_train),len(target_dataloader_train)) - 1)
        for i, (source_data, target_data) in enumerate(zip(source_dataloader_train,target_dataloader_train)):
            source_data,source_label = source_data
            target_data,_ = target_data
            source_data = source_data.cuda()
            source_label = source_label.long().cuda()
            optimizer.zero_grad()
            discr_optim.zero_grad()
            
            ##############################################
            # Train Generator
            ##############################################            
            with amp.autocast():
                discr.train_params(False)
                s_output, s_out16, s_out32 = model(source_data)
                #dom, dom16, dom32 = discr(s_output) , discr(s_out16) , discr(s_out32) # .detach() ?
                
                loss1 = loss_func(s_output, source_label.squeeze(1))
                loss2 = loss_func(s_out16, source_label.squeeze(1))
                loss3 = loss_func(s_out32, source_label.squeeze(1))
                #loss4 = discr_loss_func(dom, torch.ones(dom.shape,dtype=torch.float).cuda())
                #loss5 = discr_loss_func(dom16, torch.ones(dom.shape,dtype=torch.float).cuda())
                #loss4 = discr_loss_func(dom32, torch.ones(dom.shape,dtype=torch.float).cuda())
                t_output, t_out16, t_out32 = model(target_data)
                #dom, dom16, dom32 = discr(t_output) , discr(t_out16) , discr(t_out32)
                dom = discr(t_output)#.squeeze(1)
                #loss = loss1 + loss2 + loss3 #+ loss4 + loss5 + loss6
                loss4 = discr_loss_func(dom, torch.zeros(dom.shape,dtype=torch.float).cuda())
                #loss5 = discr_loss_func(dom16, torch.zeros(dom16.shape,dtype=torch.float).cuda().squeeze(1))
                #loss6 = discr_loss_func(dom32, torch.zeros(dom32.shape,dtype=torch.float).cuda().squeeze(1))
                loss = loss1 + loss2 + loss3 + 0.1 * loss4 #+ loss5 + loss6

                if i == image_number and epoch % 2 == 0: #saves the first image in the batch to tensorboard
                    print('epoch {}, iter {}, loss1: {}, loss2: {}, loss3: {}, loss4: {}'.format(epoch, i, loss1, loss2, loss3, loss4))
                    colorized_predictions , colorized_labels = CityScapes.visualize_prediction(s_output, source_label)
                    colorized_predictions_16 , _ = CityScapes.visualize_prediction(s_out16, source_label)
                    colorized_predictions_32 , _ = CityScapes.visualize_prediction(s_out32, source_label)

                    writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(source_data[0].cpu(),dtype='uint8'), step, dataformats='CHW')
                    writer.add_image('epoch%d/iter%d/predicted_labels_16' % (epoch, i), np.array(colorized_predictions_16), step, dataformats='HWC')
                    writer.add_image('epoch%d/iter%d/predicted_labels_32' % (epoch, i), np.array(colorized_predictions_32), step, dataformats='HWC')
                
            scaler.scale(loss).backward()

            with amp.autocast():
                ##############################################
                # Train Discriminator
                ##############################################
                discr.train_params(True)
                discr_optim.zero_grad() # is this needed?
                #dom, dom16, dom32 = discr(s_output.detach()) , discr(s_out16.detach()) , discr(s_out32.detach())
                dom = discr(s_output.detach())#.squeeze(1)
                d_loss7 = discr_loss_func(dom, torch.ones(dom.shape,dtype=torch.float).cuda())
                #d_loss8 = discr_loss_func(dom16, torch.ones(dom16.shape,dtype=torch.float).cuda().squeeze(1))
                #d_loss9 = discr_loss_func(dom32, torch.ones(dom32.shape,dtype=torch.float).cuda().squeeze(1))

                #dom, dom16, dom32 = discr(t_output.detach()) , discr(t_out16.detach()) , discr(t_out32.detach())
                dom = discr(t_output.detach())#.squeeze(1)
                d_loss7 += discr_loss_func(dom, torch.zeros(dom.shape,dtype=torch.float).cuda())
                #d_loss8 += discr_loss_func(dom16, torch.zeros(dom16.shape,dtype=torch.float).cuda().squeeze(1))
                #d_loss9 += discr_loss_func(dom32, torch.zeros(dom32.shape,dtype=torch.float).cuda().squeeze(1))
                d_loss = d_loss7 #+ d_loss8 + d_loss9

            #scaler.scale(loss).backward()
            scaler.scale(d_loss).backward()
            scaler.step(optimizer)
            scaler.step(discr_optim)
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
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, target_dataloader_val, writer, epoch, step)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
    #final evaluation
    val(args, model, target_dataloader_val, writer, epoch, step)


def train(args, model, optimizer, dataloader_train, dataloader_val, comment=''):
    #writer = SummaryWriter(comment=''.format(args.optimizer))
    writer = SummaryWriter(comment=comment)
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) #we should check if it's the right index to ignore, is it 255 or 19?
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
                output, out16, out32, _ = model(data)
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
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

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
                       default='train_da',
    )

    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='pretrained_weights\STDCNet813M_73.91.tar',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
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
    # parse.add_argument('--crop_ratio',
    #                    type=float,
    #                    default=0.5,
    #                    help='Ratio of cropped image (width and height) to input image')
    parse.add_argument('--batch_size',
                       type=int,
                       default=4, #2
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01, #0.01
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=4, #4
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,#19
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
    parse.add_argument('--comment',
                       type=str,
                       default='test',
                       help='Optional comment to add to the model name and to the log.')
    parse.add_argument('--data_transformations',
                       type=int,
                       default=0,
                       help='Select the data transformations to apply to the dataset. 0: no transformations, 1 : data augmentation')#1: random crop, 2: random crop and random horizontal flip, 3: random crop and random scale, 4: random crop, random horizontal flip and random scale.')
    return parse.parse_args()

# --dataset GTA5 --data_transformations 0 --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\test_print_features --resume False --comment test_print_features--mode train

def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes
    # to be changed
    if args.dataset == 'GTA5':
        args.crop_height, args.crop_width = 526 , 957
    
    match args.data_transformations:
        case 0:
            transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()]) #ExtRandomHorizontalFlip(),
            target_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
        case 1:
            transformations = ExtCompose([ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),ExtRandomCrop((args.crop_height, args.crop_width)), ExtToTensor()])
            target_transformations = ExtCompose([ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),ExtRandomCrop((512, 1024)), ExtToTensor()])
    eval_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
    
    if args.dataset == 'CityScapes':
        print('training on CityScapes')
        train_dataset = CityScapes(split = 'train',transforms=transformations)
        val_dataset = CityScapes(split='val',transforms=eval_transformations)

    elif args.dataset == 'GTA5':
        print('training on GTA5')
        train_dataset = GTA5(root='dataset',split="train",transforms=transformations)
        val_dataset = GTA5(root='dataset',split="eval",transforms=eval_transformations)

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on GTA and validating on Cityscapes')
        train_dataset = GTA5(root='dataset',transforms=transformations)
        target_dataset_train = CityScapes(split = 'train',transforms=transformations)
        val_dataset = CityScapes(split = 'val',transforms=eval_transformations)
    else:
        print('not supported dataset \n')
        return None
    
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
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    
    
    if args.resume or args.mode == 'test':
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
    
    if args.comment == '':
        args.comment = "_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate)

    match args.mode:
        case 'train':
            ## train loop
            train(args, model, optimizer, source_dataloader_train, dataloader_val, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case 'train_da':
            train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment)
        case 'test':
            writer = SummaryWriter(comment=args.comment)
            val(args, model, dataloader_val,writer=writer,epoch=0,step=0)
        case _:
            print('not supported mode \n')
            return None
if __name__ == "__main__":
    main()