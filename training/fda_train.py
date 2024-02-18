# LIBRARIES
import torch
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter

# PERSONAL
# Models
# Datasets
from datasets.cityscapes import CityScapes
# Utils
from utils.general import poly_lr_scheduler, save_ckpt, load_ckpt
from utils.fda import FDA_source_to_target, EntropyMinimizationLoss
from eval import evaluate_and_save_model
from utils.aug import V2Normalize
# GLOBAL VARIABLES
# Image mean of the Cityscapes dataset (used for normalization)
MEAN = torch.tensor([0.4079, 0.4575, 0.4811])
STD = torch.tensor([1.0, 1.0, 1.0])
# COMMAND LINE
# python main.py --mode train_fda --dataset CROSS_DOMAIN --save_model_path trained_models\fda_0.09 --comment fda_0.09 --data_transformation 0 --batch_size 5 --num_workers 4 --optimizer adam

def train_fda(args, model, optimizer, source_dataloader_train, target_dataloader_train, target_dataloader_val, comment='', starting_epoch=0,
              beta=0.09, ent_weight=0.005, ita=2.0):
    """
    Train the model using `Fourier Domain Adaptation (DA)` for semantic segmentation tasks. 
    
    This function trains the model using the source and target domain data, 
    and adapts the appearance of the source images to the target domain using FDA.
    
    Args:
    - ...
    - beta: hyperparameter to control the size of the low-frequency window to be swapped (float)
    - ent_weight: weight for the entropy minimization loss (float)
    - ita: coefficient for the robust norm on entropy minimization loss in Charbonnier penality (float)
    """

    # 1 Parameters Initialization
    writer = SummaryWriter(comment=comment) # Tensorboard writer
    scaler = amp.GradScaler() # Automatic Mixed Precision
    max_miou = 0
    step = 0

    # 2. Resume Model from Checkpoint
    if args.resume:
        try:
            max_miou, starting_epoch = load_ckpt(args, optimizer=optimizer, model=model)
            print('successfully resume model from %s' % args.resume_model_path)
        except Exception as e:
            print(e)
            print('resume failed, try again')
            return None

    # 3. Loss Functions
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    ent_loss = EntropyMinimizationLoss()
    
    # 4. Training Loop for each epoch
    for epoch in range(starting_epoch,args.num_epochs):

        # 4.1. Adjust Learning Rate
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)

        # 4.2. Set up the model to train mode
        model.train()
        tq = tqdm(total= min(len(source_dataloader_train),len(target_dataloader_train)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = [] # Loss for each batch

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

            # 4.5.4. Apply FDA
            
            #############
            # Apply FDA #
            #############

            # FDA.1. Apply FDA to the source images to adapt their appearance to the target domain
            original_source_data = source_data.clone()
            t_source_data = FDA_source_to_target(source_data, target_data, beta)

            # FDA.2. Normalize the source and target images and rescale them to the range [0, 1]
            t_source_data = t_source_data / 255.0
            t_source_data, _ = V2Normalize(mean=MEAN,std=STD)(t_source_data,lbl=source_label)
            target_data = target_data / 255.0
            target_data, _ = V2Normalize(mean=MEAN,std=STD)(target_data, lbl=source_label)

            # FDA.3. Get the predictions for the source images
            with amp.autocast():
                
                # FDA.3.1. Forward pass -> multi-scale outputs
                s_output, _, _ = model(t_source_data)
                
                # FDA.3.2. Compute the segmentation loss for the source domain
                ce_loss_value = ce_loss(s_output, source_label.squeeze(1))
            
            # FDA.4. Get the predictions for the target images
            with amp.autocast():
                
                # FDA.4.1. Forward pass -> multi-scale outputs
                t_output, _, _ = model(target_data)

                # FDA.4.2. Compute the entropy minimization loss for the target domain
                ent_loss_value = ent_loss(t_output, ita)
                
                # FDA.4.3. Compute the total loss for the batch
                loss = ce_loss_value + ent_weight * ent_loss_value

            # FDA.5. Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #############

            # 4.5.4. Save the randomly selected image in the batch to tensorboard
            if i == image_number and epoch % 2 == 0: #saves the first image in the batch to tensorboard
                print('epoch {}, iter {}, tot_loss: {}'.format(epoch, i, loss))
                colorized_predictions , colorized_labels = CityScapes.visualize_prediction(s_output, source_label.squeeze(1))
                writer.add_image('epoch%d/iter%d/predicted_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('epoch%d/iter%d/correct_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('epoch%d/iter%d/original_data' % (epoch, i), np.array(original_source_data[0].detach().cpu(),dtype='uint8'), step, dataformats='CHW')
                t_source_data_vis = ((t_source_data + MEAN[:, None, None].cuda()) * 255)
                writer.add_image('epoch%d/iter%d/stylized_data' % (epoch, i), np.array(t_source_data_vis[0].detach().cpu(),dtype='uint8'), step, dataformats='CHW')

            # 4.5.5. Update the progress bar
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            
            # 4.5.6. Save the loss for the batch
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()

        # 4.6. Save the average loss for the epoch
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        # 4.7. Save a checkpoint of the model every {args.checkpoint_step} epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_ckpt(args=args,model=model, optimizer=optimizer,cur_epoch=epoch,best_score= max_miou,name='latest.pth')
        
        # 4.8. Evaluate the model on the validation set every {args.validation_step} epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            max_miou = evaluate_and_save_model(args, model, target_dataloader_val, writer, epoch, step, max_miou, mean=MEAN, std=STD)
    
    # 5. Final Evaluation
    max_miou = evaluate_and_save_model(args, model, target_dataloader_val, writer, epoch, step, max_miou, mean=MEAN, std=STD)
