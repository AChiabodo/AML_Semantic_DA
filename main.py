#!/usr/bin/python
# -*- encoding: utf-8 -*-

# LIBRARIES
import torch
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import random
from tensorboardX import SummaryWriter
# PERSONAL
# Models
from model.model_stages import BiSeNet
# Datasets
from datasets.cityscapes import CityScapes
from datasets.GTA5 import GTA5
# Utils
from utils.general import str2bool, load_ckpt
from utils.aug import ExtCompose, ExtRandomHorizontalFlip , ExtScale , ExtRandomCrop, ExtGaussianBlur, ExtColorJitter, ExtRandomCompose, ExtResize, ExtToV2Tensor, V2Normalize
from utils.fda import test_mbt, save_pseudo
# Training
from training.simple_train import train
from training.single_layer_da_train import train_da
from training.fda_train import train_fda
from training.fda_self_learning_train import train_self_learning_fda
# Evaluation
from eval import val

# GLOBAL VARIABLES
# Image mean of the ImageNet dataset (used for normalization)
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
# Image mean of the Cityscapes dataset (used for normalization in FDA)
MEAN_CS = torch.tensor([0.4079, 0.4575, 0.4811])
STD_CS = torch.tensor([1.0, 1.0, 1.0])

def parse_args():
    """Parse input arguments from command line"""
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='save_pseudo',
                       help='Select between simple training (train),'+
                            'training with Domain Adaptation (train_da),'+
                            'training with Fourier Domain Adaptation (train_fda),'+
                            'FDA adapted with Multi-band Transfer (test_mbt)'+
                            'saving the pseudo labels generated using MBT (save_pseudo)'+
                            'training with self-learning and pseudo labels (self_learning)'+
                            'or testing an already trained model (test)'
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
                       default=5, #2
                       help='Number of images in each batch'
    )
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.001, #0.01
                        help='learning rate used during training to adjust the weights of the model'
    )
    parse.add_argument('--num_workers',
                       type=int,
                       default=2,
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
                       help='Define the path to the model that should be loaded for testing or resuming its training. If void, the best model trained so far will be loaded'
    )
    parse.add_argument('--dataset',
                          type=str,
                          default='CROSS_DOMAIN',
                          help='CityScapes, GTA5 or CROSS_DOMAIN. Define on which dataset the model should be trained and evaluated.'
    )
    parse.add_argument('--comment',
                       type=str,
                       default='test_default',
                       help='Comment to add to the log and on tensorboard to identify the model'
    )
    parse.add_argument('--data_transformations',
                       type=int,
                       default=0,
                       help='Select transformations to be applied on the dataset images (0: no transformations, 1 : data augmentation)'
    )
    parse.add_argument('--beta',
                       type=float,
                       default=0.05,
                       help='Select beta for the Fourier Domain Adaptation'
    )
    parse.add_argument('--d_lr',
                       type=float,
                       default=0.001,
                       help='Select learning rate for the Domain Discriminator'
    )
    parse.add_argument('--fda_b1_path',
                        type=str,
                        default='trained_models\\augm_selflearn_fda0.01\\best.pth',
                        help='Path to the model trained with beta=0.01'
    )
    parse.add_argument('--fda_b2_path',
                        type=str,
                        default='trained_models\\augm_selflearn_fda0.05\\best.pth',
                        help='Path to the model trained with beta=0.05'
    )
    parse.add_argument('--fda_b3_path',
                        type=str,
                        default='trained_models\\augm_selflearn_fda0.09\\best.pth',
                        help='Path to the model trained with beta=0.09'
    )
    parse.add_argument('--save_pseudo_path',
                        type=str,
                        default='dataset\\Cityscapes\\pseudo_label',
                        help='Path to the folder where to save the pseudo labels generated using Multi-band Transfer'
    )
    return parse.parse_args()

def main():

    # 1. Initialization
    args = parse_args()
    n_classes = args.num_classes
    args.dataset = args.dataset.upper()
    
    # 2. Data Transformations Selection
    size = (512,1024)
    if args.dataset == 'GTA5':
        size = (526,957)
        

    """By default, the images are resized to 0.5 of their original size to reduce computational cost"""
    standard_transformations = ExtCompose([ExtResize(size=size), ExtToV2Tensor(), V2Normalize(scale=True)])
    """The Validation Set is also resized to 0.5 of its original size"""
    eval_transformations = standard_transformations

    match args.data_transformations:
        case 0:
            """
            No Data Augmentation
            - Images are resized to 0.5 of their original size to reduce computational cost
            - No extra transformations are applied to the dataset
            - For FDA, GTA5 images are resized to half the size of the Cityscapes images
            """
            transformations = standard_transformations
            target_transformations = standard_transformations
            
        case 1:
            """
            Feeble Data Augmentation -> 50% probability to be applied
            - Images from GTA5 are first enlarged and then cropped to the half of their original size
                1. To pay attention to the details of the images
                2. To reduce computational cost
            - Images are randomly flipped horizontally
            """
            transformations = ExtRandomCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomCrop(size),
                ExtRandomHorizontalFlip(),
                ExtToV2Tensor(),
                V2Normalize(scale=True)
                ],
                [ExtResize(size), ExtToV2Tensor(),V2Normalize(scale=True)])
            target_transformations = ExtCompose([ExtResize(size), ExtToV2Tensor(),V2Normalize(scale=True)])
            eval_transformations = target_transformations

        case 2:
            """
            Precise Data Augmentation -> 50% probability to be applied
            - Images from GTA5 are first enlarged and then cropped to the half of their original size
                1. To pay attention to the details of the images
                2. To reduce computational cost
            - Images are randomly flipped horizontally
            - Images are randomly blurred
            - Images are randomly color jittered
            """
            transformations = ExtRandomCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomCrop(size),
                ExtRandomHorizontalFlip(),
                ExtGaussianBlur(p=0.5, radius=1),
                ExtColorJitter(p=0.5, brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
                ExtToV2Tensor(),
                V2Normalize(mean=MEAN,std=STD, scale=True)],
                [ExtResize(size), ExtToV2Tensor(),V2Normalize(mean=MEAN,std=STD, scale=True)])
            target_transformations = transformations

    if args.mode == 'save_pseudo' or args.mode == 'test_mbt' :
                transformations = ExtCompose([ExtResize(size), ExtToV2Tensor(), V2Normalize(scale=True,mean=MEAN_CS,std=STD_CS)])
                target_transformations = ExtCompose([ExtResize(size), ExtToV2Tensor(),V2Normalize(scale=True,mean=MEAN_CS,std=STD_CS)])
                eval_transformations = ExtCompose([ExtResize(size), ExtToV2Tensor(),V2Normalize(scale=True,mean=MEAN_CS,std=STD_CS)])

    if args.mode == 'train_fda' or args.mode == 'self_learning':
                print('Applying FDA specific transformations')
                transformations = ExtRandomCompose([
                ExtScale(random.choice([0.75,1,1.25,1.5,1.75,2]),interpolation=Image.Resampling.BILINEAR),
                ExtRandomCrop(size),
                ExtRandomHorizontalFlip(),
                ExtToV2Tensor()],
                [ExtResize(size), ExtToV2Tensor()])
                target_transformations = ExtCompose([ExtResize(size), ExtToV2Tensor()])
                if args.mode == 'self_learning':
                    target_transformations = transformations
                eval_transformations = ExtCompose([ExtResize(size), ExtToV2Tensor(),V2Normalize(mean=MEAN_CS,std=STD_CS)])
    
    # 3. Datasets Selection
    
    if args.dataset == 'CITYSCAPES':
        print('training on CityScapes')
        train_dataset = CityScapes(split='train',transforms=transformations)
        val_dataset = CityScapes(split='val',transforms=eval_transformations)

    elif args.dataset == 'GTA5':
        print('training on GTA5')
        train_dataset = GTA5(root='dataset',split="train",transforms=transformations)
        val_dataset = GTA5(root='dataset',split="eval",transforms=eval_transformations)

    elif args.dataset == 'CROSS_DOMAIN':
        print('training on GTA and validating on Cityscapes')
        train_dataset = GTA5(root='dataset',transforms=transformations)
        target_dataset_train = CityScapes(split='train',transforms=target_transformations)
        if args.mode == 'self_learning':
            target_dataset_train = CityScapes(split='train',mode='pseudo',transforms=target_transformations)
        val_dataset = CityScapes(split='val',transforms=eval_transformations)
    else:
        print('Dataset not supported \n')
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

    # 6. GPU Parallelization
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # 7. Optimizer Selection
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

    # 8. Comment for Tensorboard
    if args.comment == '':
        args.comment = "_{}_{}_{}_{}".format(args.mode,args.dataset,args.batch_size,args.learning_rate)

    # 9. Start Training or Evaluation
    match args.mode:
        case 'train':
            # 9.1. Simple Training on Source Dataset
            train(args, model, optimizer, source_dataloader_train, dataloader_val, comment=args.comment)
        case 'train_da':
            # 9.2. Training with Domain Adaptation
            train_da(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment, d_lr=args.d_lr)
        case 'train_fda':
            # 9.3. Training with Fourier Domain Adaptation
            train_fda(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment, beta=args.beta)
        case 'test_mbt':
            # 9.4. Testing the already trained FDA models using Multi-band Transfer => MBT on the Validation Set of Cityscapes
            test_mbt(args, dataloader_val, path_b1=args.fda_b1_path, path_b2=args.fda_b2_path, path_b3=args.fda_b3_path)
        case 'save_pseudo':
            # 9.5. Save the pseudo labels generated using Multi-band Transfer => MBT on the Training Set of Cityscapes
            save_pseudo(args, target_dataloader_train, path_b1=args.fda_b1_path, path_b2=args.fda_b2_path, path_b3=args.fda_b3_path, save_path=args.save_pseudo_path)
        case 'self_learning':
            # 9.6. Training with Self-Learning FDA and Pseudo Labels
            train_self_learning_fda(args, model, optimizer, source_dataloader_train, target_dataloader_train, dataloader_val, comment=args.comment, beta=args.beta)
        case 'test':
            # 9.7. Load the trained model and evaluate it on the Validation Set
            try:
                load_ckpt(args, model=model)
                print('successfully resume model from %s' % args.resume_model_path)
            except Exception as e:
                print(e)
                print('resume failed, try again')
                return None
            
            writer = SummaryWriter(comment=args.comment)
            val(args, model, dataloader_val,writer=writer,epoch=0,step=0)
        case _:
            print('not supported mode \n')
            return None

if __name__ == "__main__":
    main()