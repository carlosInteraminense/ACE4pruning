import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
from torch_lr_finder import LRFinder

import vgg


data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

data_dir = '/media/revai/Data/Carlos/imagenet/50_classes/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
											 shuffle=True, num_workers=12)
			  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfgs = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	device = 'cuda'
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_ephoc = 0

	n = 1
	n_it = 1000
	n_time = time.time()
	
	fout = open('log_50_classes_finetuning.txt', 'w')
	for epoch in range(0, num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# print (inputs)
					#print ('{}. oi'.format(n))
					# print ('------')
					# print ('------')
					outputs = model(inputs)[-1]
					_, preds = torch.max(outputs, 1)
					# print (preds)
					# print (labels)
					# input()
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				if (n%100 == 0):
					time_elapsed = n_time - time.time()
					print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
					print('{}. {} Loss: {:.4f}'.format(n, phase, loss.item()))
					print ('----')
					fout.write('Time elapsed {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
					fout.write('{}. {} Loss: {:.4f}\n'.format(n, phase, loss.item()))
					fout.write('----\n')
					n_time = time.time()
				n+=1
				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			if phase == 'train':
				scheduler.step(epoch_loss)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))
			
			fout.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format( phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_ephoc = epoch
				best_model_wts = copy.deepcopy(model.state_dict())
				
		torch.save(model, './snapshots_50_sgd_finetuning/model_weights_epoch_{}.pth'.format(epoch))
		time_elapsed = time.time() - since
		print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		fout.write('Time elapsed {:.0f}m {:.0f}s\n\n'.format(time_elapsed // 60, time_elapsed % 60))
		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	fout.write('Training complete in {:.0f}m {:.0f}s\n'.format( time_elapsed // 60, time_elapsed % 60))
	fout.write('Best val Acc: {:4f}, epoch: {}\n'.format(best_acc, best_ephoc))
	fout.close()
	# load best model weights
	model.load_state_dict(best_model_wts)
	torch.save(model, './snapshots_50_sgd_freezed_conv_layers/best_model.pth')
	return model


#model = vgg.VGG(make_layers(cfgs['D'], batch_norm=False), num_classes=50, init_weights=True)


vgg16 = models.vgg16(pretrained=True)
model = vgg.VGG(make_layers(cfgs['D'], batch_norm=False), num_classes=50, init_weights=True)

d_aux={
	'classifier.0.weight':'hl_1.weight',
	'classifier.0.bias':'hl_1.bias',
	'classifier.3.weight':'hl_2.weight',
	'classifier.3.bias':'hl_2.bias',
	'classifier.6.weight':'out_layer.weight',
	'classifier.6.bias':'out_layer.bias'}

def copy_bias(bias, bias_new):
	for i in range(len(bias)):
		bias_new[i] = bias[i]

state_dict_pre_trained = vgg16.state_dict()
state_dict = model.state_dict()
for key in state_dict_pre_trained:
	if ('features' not in key): continue
	#print (key)
	w_pre = state_dict_pre_trained[key].data.numpy()	
	if (key in state_dict):
		w_new = state_dict[key].data.numpy()
	else:
		w_new = state_dict[d_aux[key]].data.numpy()

	copy_bias(w_pre, w_new)


######
#model = torch.load('/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/model_weights_epoch_200.pth')
######
# Here the size of each output sample is set to 2.
# for layer_name, param in zip(model.state_dict(), model.parameters()):
# 	if ('features' not in layer_name): continue
# 	param.requires_grad = False
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).



model = model.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(model.parameters(), lr=0.0288, momentum=0.9)
#optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# lr_finder = LRFinder(model, optimizer_ft, criterion, device="cuda")
# lr_finder.range_test(dataloaders['train'], val_loader=dataloaders['val'], end_lr=1, num_iter=2000)
# lr_finder.plot(log_lr=False)
# plt.savefig('lrsugestion.png')
# lr_finder.reset()

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=300)