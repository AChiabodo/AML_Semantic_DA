# LIBRARIES
import torch
import numpy as np
import random
import os

# PERSONAL
# Datasets
from datasets.cityscapes import CityScapes
# Utils
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, save_ckpt


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

    with torch.no_grad(): # No need to track the gradients during validation

        # 1. Initialization of evaluation metrics
        model.eval()
        precision_record = [] # List of precision per pixel for each image
        hist = np.zeros((args.num_classes, args.num_classes)) # Confusion Matrix for mIoU

        # 2. Select a random image to save to Tensorboard
        random_sample = random.randint(0, len(dataloader) - 1)

        # 3. Validation Loop
        for i, (data, label) in enumerate(dataloader):

            # 3.1. Load data and label to GPU
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # 3.2. Forward pass -> original scale output
            predict, _, _ = model(data)
            
            # 3.3. Save the randomly selected image to Tensorboard
            if i == random_sample and writer is not None:
                colorized_predictions , colorized_labels = CityScapes.visualize_prediction(predict, label)
                writer.add_image('eval%d/iter%d/predicted_eval_labels' % (epoch, i), np.array(colorized_predictions), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/correct_eval_labels' % (epoch, i), np.array(colorized_labels), step, dataformats='HWC')
                writer.add_image('eval%d/iter%d/eval_original _data' % (epoch, i), np.array(data[0].cpu(),dtype='uint8'), step, dataformats='CHW')

            # 3.4. Get the predicted label
            predict = predict.squeeze(0) # Squash batch dimension
            predict = reverse_one_hot(predict) # Convert to 2D tensor where each pixel is the class index
            predict = np.array(predict.cpu()) # Convert to numpy array

            # 3.5. Get the ground truth label
            label = label.squeeze() # Squash batch dimension
            label = np.array(label.cpu()) # Convert to numpy array

            # 3.6. Compute precision per pixel and update the confusion matrix
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
            
        # 4. Compute metrics
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def evaluate_and_save_model(args, model, dataloader_val, writer, epoch, step, max_miou):
    """
    Evaluate the model and save it if performance has improved.

    Args:
    - ...
    - max_miou: Maximum mean Intersection over Union achieved so far.

    Returns:
    - Updated max_miou after evaluation.
    """
    precision, miou = val(args, model, dataloader_val, writer, epoch, step)
    if miou > max_miou:
        max_miou = miou
        os.makedirs(args.save_model_path, exist_ok=True)
        #torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
        save_ckpt(args=args,model=model, optimizer=None,cur_epoch=epoch,best_score= max_miou,name='best.pth',discriminator_optimizer=None,discriminator=None)

    writer.add_scalar('epoch/precision_val', precision, epoch)
    writer.add_scalar('epoch/miou_val', miou, epoch)

    return max_miou
