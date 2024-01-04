#!/usr/bin/python
# -*- encoding: utf-8 -*-

# LIBRARIES
import torch
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import random
import os
from tensorboardX import SummaryWriter

# PERSONAL
# Models
from model.model_stages import BiSeNet
# Datasets
from datasets.cityscapes import CityScapes
from datasets.GTA5 import GTA5
# Utils
from utils import str2bool
from training.data_augmentation import ExtCompose, ExtToTensor, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop, ExtGaussianBlur, ExtColorJitter
from training.simple_train import train
from training.single_layer_da_train import train_da
from eval import val


"""
  LAST TRAINING TRIALS:

  --dataset GTA5 --data_transformations 0 --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\test_print_features --resume False --comment test_print_features--mode train
  
  --mode train_da --dataset CROSS_DOMAIN --save_model_path trained_models\adv_single_layer_lam0.001_softmax_resumed --comment adv_single_layer_lam0.005_softmax --data_transformation 0 --batch_size 4 --learning_rate 0.002 --num_workers 4 --optimizer sgd --resume True --resume_model_path trained_models\avd_single_layer_lam0.005_softmax\best.pth

  & C:/Users/aless/Documents/Codice/AML_Semantic_DA/.venv/Scripts/python.exe c:/Users/aless/Documents/Codice/AML_Semantic_DA/train.py 
  --mode train --dataset CROSS_DOMAIN --save_model_path trained_models\bea_data_augm_test --comment bea_data_augm_test --data_transformation 2 --batch_size 5 --num_workers 4 --optimizer adam --crop_height 526 --crop_width 957
"""

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

def main():

    # 1. Initialization
    args = parse_args()
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    #TODO: to be changed
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
            Feeble Data Augmentation
            - Images from GTA5 are first enlarged and then cropped to the half of their original size
                1. To pay attention to the details of the images
                2. To reduce computational cost
            - Images are randomly flipped horizontally
            """
            transformations = ExtCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomCrop((args.crop_height, args.crop_width)),
                ExtRandomHorizontalFlip(),
                ExtToTensor()])
            target_transformations = ExtCompose([ExtScale(0.5,interpolation=Image.Resampling.BILINEAR), ExtToTensor()])
        case 2:
            """
            Precise Data Augmentation
            - Images from GTA5 are first enlarged and then cropped to the half of their original size
                1. To pay attention to the details of the images
                2. To reduce computational cost
            - Images are randomly flipped horizontally
            - Images are randomly blurred
            - Images are randomly color jittered
            """
            transformations = ExtCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomCrop((args.crop_height, args.crop_width)),
                ExtRandomHorizontalFlip(),
                ExtGaussianBlur(p=0.5, radius=1),
                ExtColorJitter(p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
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

    # 10. Path to Pretrained Weights
    if os.name == 'nt':
        args.pretrain_path = args.pretrain_path.replace('\\','/')

    # 11. Start Training or Evaluation
    match args.mode:
        case 'train':
            # 11.1. Simple Training on Source Dataset
            train(args, model, optimizer, source_dataloader_train, dataloader_val, comment="_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate))
        case 'train_da':
            # 11.2. Training with Domain Adaptation
            train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment)
        case 'test':
            # 11.3. Evaluation of an already trained model on the Validation Set
            writer = SummaryWriter(comment=args.comment)
            val(args, model, dataloader_val,writer=writer,epoch=0,step=0)
        case _:
            print('not supported mode \n')
            return None

if __name__ == "__main__":
    main()