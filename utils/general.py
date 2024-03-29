import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import random
import numbers
import torchvision
import argparse
import os

# Polynomial learning rate scheduler
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
	# if iter % lr_decay_iter or iter > max_iter:
	# 	return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr

# Get label information from a CSV file
def get_label_info(csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	label = {}
	for iter, row in ann.iterrows():
		label_name = row['name']
		r = row['r']
		g = row['g']
		b = row['b']
		class_11 = row['class_11']
		label[label_name] = [int(r), int(g), int(b), class_11]
	return label

# Convert labeled image to one-hot encoded semantic map
def one_hot_it(label, label_info):
	# return semantic_map -> [H, W]
	semantic_map = np.zeros(label.shape[:-1])
	for index, info in enumerate(label_info):
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map[class_map] = index
		# semantic_map.append(class_map)
	# semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map

# Convert labeled image to one-hot encoded semantic map with void class
def one_hot_it_v11(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = np.zeros(label.shape[:-1])
	# from 0 to 11, and 11 means void
	class_index = 0
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map[class_map] = class_index
			class_index += 1
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			semantic_map[class_map] = 11
	return semantic_map

# Convert labeled image to one-hot encoded semantic map with void class (for dice loss)
def one_hot_it_v11_dice(label, label_info):
	# return semantic_map -> [H, W, class_num]
	semantic_map = []
	void = np.zeros(label.shape[:2])
	for index, info in enumerate(label_info):
		color = label_info[info][:3]
		class_11 = label_info[info][3]
		if class_11 == 1:
			# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			# semantic_map[class_map] = index
			semantic_map.append(class_map)
		else:
			equality = np.equal(label, color)
			class_map = np.all(equality, axis=-1)
			void[class_map] = 1
	semantic_map.append(void)
	semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
	return semantic_map

# Reverse one-hot encoding to get class indices
def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x

# Color code the segmentation results
def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
	label_values.append([0, 0, 0])
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]
	return x

# Compute global accuracy
def compute_global_accuracy(pred, label):
	pred = pred.flatten()
	label = label.flatten()
	total = len(label)
	count = 0.0
	for i in range(total):
		if pred[i] == label[i]:
			count = count + 1.0
	return float(count) / float(total)

# Fast histogram computation
def fast_hist(a, b, n):
	'''
	a and b are predict and mask respectively
	n is the number of classes
	'''
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# Per-class intersection over union
def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


# Random crop transformation
class RandomCrop(object):
	"""Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

	def __init__(self, size, seed, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.seed = seed

	@staticmethod
	def get_params(img, output_size, seed):
		"""Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		random.seed(seed)
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = torchvision.transforms.functional.pad(img, self.padding)

		# pad the width if needed
		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
		# pad the height if needed
		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.seed)

		return torchvision.transforms.functional.crop(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


# Calculate mean intersection over union
def cal_miou(miou_list, csv_path):
	# return label -> {label_name: [r_value, g_value, b_value, ...}
	ann = pd.read_csv(csv_path)
	miou_dict = {}
	cnt = 0
	for iter, row in ann.iterrows():
		label_name = row['name']
		class_11 = int(row['class_11'])
		if class_11 == 1:
			miou_dict[label_name] = miou_list[cnt]
			cnt += 1
	return miou_dict, np.mean(miou_list)

# Online Hard Example Mining CrossEntropy Loss
class OHEM_CrossEntroy_Loss(nn.Module):
	def __init__(self, threshold, keep_num):
		super(OHEM_CrossEntroy_Loss, self).__init__()
		self.threshold = threshold
		self.keep_num = keep_num
		self.loss_function = nn.CrossEntropyLoss(reduction='none')

	def forward(self, output, target):
		loss = self.loss_function(output, target).view(-1)
		loss, loss_index = torch.sort(loss, descending=True)
		threshold_in_keep_num = loss[self.keep_num]
		if threshold_in_keep_num > self.threshold:
			loss = loss[loss>self.threshold]
		else:
			loss = loss[:self.keep_num]
		return torch.mean(loss)

# Group weights for optimization with weight decay
def group_weight(weight_group, module, norm_layer, lr):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)

	assert len(list(module.parameters())) == len(group_decay) + len(
		group_no_decay)
	weight_group.append(dict(params=group_decay, lr=lr))
	weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
	return weight_group

# Convert string to boolean
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')
 
# Save model checkpoint
def save_ckpt(args,model, best_score, cur_epoch, optimizer= None, discriminator=None, cur_itrs=None, discriminator_optimizer=None, name=None):
	
	# 1. Create the directory if it doesn't exist
	if not os.path.exists(args.save_model_path):
		os.makedirs(args.save_model_path)
	
	# 2. Define the checkpoint filename
	if name is None:
		checkpoint_filename = 'epoch_{}_{}.pth'.format(cur_epoch, args.comment)
	else:
		checkpoint_filename = name

	# 2.1 Define the checkpoint path
	checkpoint_path = os.path.join(args.save_model_path, checkpoint_filename)

	# 3. Save the checkpoint
	torch.save({
        "cur_epoch": cur_epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_score": best_score,
        "discriminator": discriminator.module.state_dict() if discriminator is not None else None,
        "cur_itrs": cur_itrs,  # aggiungi questa linea se cur_itrs è importante
		"discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None
    }, checkpoint_path)

	# 4. Print the checkpoint path
	print("Model saved as %s" % checkpoint_path)

# Load model checkpoint
def load_ckpt(args, model, optimizer = None, discriminator=None, discriminator_optimizer=None, verbose=False) -> (float, int):
	best_score = 0
	# 1. Check if the model path is specified, otherwise load the best model trained so far
	if args.resume_model_path == '':
		args.resume_model_path = os.path.join(args.save_model_path, 'best.pth')
		print('No model path specified. Loading the best model trained so far: {}'.format(args.resume_model_path))

	# 2. Load the checkpoint
	checkpoint = torch.load(args.resume_model_path)

	# 3. Load the model state dict
	model.module.load_state_dict(checkpoint["model_state_dict"])
	if verbose : 
		print("Model restored from %s" % args.resume_model_path)
	
	# 4. If we're resuming training, also load the optimizer state dict, the current epoch and the best score so far
	if args.resume:
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) if optimizer is not None else None
		if verbose :
			print("Optimizer restored from %s" % args.resume_model_path)
		cur_epoch = checkpoint["cur_epoch"]  # carica l'epoca corrente
		best_score = checkpoint['best_score']
		
		# 4.1. Load the discriminator and its optimizer if it was a domain adaptation task
		if discriminator is not None and discriminator_optimizer is not None and 'discriminator' in checkpoint and 'discriminator_optimizer' in checkpoint:
			discriminator.module.load_state_dict(checkpoint['discriminator'])
			discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
			if verbose :
				print("Discriminator restored from %s" % args.resume_model_path)
		print("Training state correctly restored from %s" % args.resume_model_path)
	else:
		cur_epoch = 0
	# 5. Return the best score and the current epoch
	return best_score, cur_epoch

from PIL import Image
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes

def save_predictions(args, target_dataloader_train,
							path_cross = 'trained_models\\cross\\best.pth',
              path_aug = 'trained_models\\aug\\best.pth',
							path_aug_da = 'trained_models\\aug_da\\best.pth',
							path_mbt1 = 'trained_models\\mbt1\\best.pth',
							path_mbt2 = 'trained_models\\mbt2\\best.pth',
              save_path = 'dataset\\predictions'
            ):

	with torch.no_grad(): # No need to track the gradients during validation

		# Initialization
		backbone = args.backbone
		n_classes = args.num_classes
		pretrain_path = args.pretrain_path
		use_conv_last = args.use_conv_last
				
		# Create the folder to save the pseudo-labels
		if not os.path.exists(save_path):
				os.makedirs(save_path)

		# Initialize the arrays to store the pseudo-labels
		predicted_labels = [] #np.zeros((len(target_dataloader_train), 512, 1024), dtype=np.uint8)
		predicted_probs = [] #np.zeros((len(target_dataloader_train), 512, 1024), dtype=np.float32)
		image_names = []

		print('Pseudo-labels will be saved in: ', save_path)

		# Load the models trained with different betas
		checkpoint_b1 = torch.load(path_cross)   
		model1 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
		model1.load_state_dict(checkpoint_b1['model_state_dict'])
		model1.cuda()
		model1.eval()

		checkpoint_b2 = torch.load(path_aug)
		model2 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
		model2.load_state_dict(checkpoint_b2['model_state_dict'])
		model2.cuda()
		model2.eval()

		checkpoint_b3 = torch.load(path_aug_da)
		model3 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
		model3.load_state_dict(checkpoint_b3['model_state_dict'])
		model3.cuda()
		model3.eval()

		checkpoint_b4 = torch.load(path_mbt1)
		model4 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
		model4.load_state_dict(checkpoint_b4['model_state_dict'])
		model4.cuda()
		model4.eval()

		checkpoint_b5 = torch.load(path_mbt2)
		model5 = BiSeNet(backbone, n_classes, pretrain_path, use_conv_last)
		model5.load_state_dict(checkpoint_b5['model_state_dict'])
		model5.cuda()
		model5.eval()

		print('Models loaded!')

		# Iterate over the validation dataset
		for i, (data, label) in enumerate(target_dataloader_train):

			if i > 2:
				break
			
			# Load data and label to GPU
			label = label.type(torch.LongTensor)
			data = data.cuda()
			label = label.long().cuda()

			# Forward pass -> original scale output
			predict1, _, _ = model1(data)
			predict2, _, _ = model2(data)
			predict3, _, _ = model3(data)
			predict4, _, _ = model4(data)
			predict5, _, _ = model5(data)
          
			# For each image in the batch save the predicted label image
			for j in range(predict1.size(0)):

				# Get the image name
				path = target_dataloader_train.dataset.images[i*args.batch_size + j]
				name = path.split('\\')[-1]
				name = name.split('_leftImg8bit.png')[0] + '_predicted_label'
				image_names.append(save_path + '\\' + name)
				
				# Get the predicted labels
				output1 = predict1[j].cpu().numpy()
				output1 = output1.transpose(1, 2, 0)
				output1 = np.asarray(np.argmax(output1, axis=2), dtype=np.uint8)
				name1 = name + '_cross.png'
				
				output2 = predict2[j].cpu().numpy()
				output2 = output2.transpose(1, 2, 0)
				output2 = np.asarray(np.argmax(output2, axis=2), dtype=np.uint8)
				name2 = name + '_aug.png'

				output3 = predict3[j].cpu().numpy()
				output3 = output3.transpose(1, 2, 0)
				output3 = np.asarray(np.argmax(output3, axis=2), dtype=np.uint8)
				name3 = name + '_aug_da.png'

				output4 = predict4[j].cpu().numpy()
				output4 = output4.transpose(1, 2, 0)
				output4 = np.asarray(np.argmax(output4, axis=2), dtype=np.uint8)
				name4 = name + '_mbt1.png'

				output5 = predict5[j].cpu().numpy()
				output5 = output5.transpose(1, 2, 0)
				output5 = np.asarray(np.argmax(output5, axis=2), dtype=np.uint8)
				name5 = name + '_mbt2.png'

				# Save the images in the folder
				for output, name in zip([output1, output2, output3, output4, output5], [name1, name2, name3, name4, name5]):
					output_col = CityScapes.decode_target(output)
					output_im = Image.fromarray(output_col.astype(np.uint8))
					output_im.save(os.path.join(save_path, name))
    
		print('Prediction saved!')