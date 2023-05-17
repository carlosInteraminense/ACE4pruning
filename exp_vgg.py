import torch
import torch.nn as nn
import torch_pruning as tp
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision import transforms, datasets
import numpy as np
import torch.nn.utils.prune as prune

from decimal import *

from PIL import Image
import math
import vgg
import os
import timeit
import operator
import copy
import time

device = 'cuda'
N_CLASSES = 50
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def copy_bias(bias, bias_new):
	for i in range(len(bias)):
		bias_new[i] = bias[i]

cfgs = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
											 shuffle=True, num_workers=12)
			  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, log_filename='log.txt', dir_snapshot_out='snapshots/'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	for layer_name, param in zip(model.state_dict(), model.parameters()):
		if ('features' not in layer_name): continue
		param.requires_grad = False
		#print ('ALGUM DEU FALSE. .AINDA BEEEEMM!!!!')
	
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
	since = time.time()
	model.cuda()
	if torch.cuda.is_available():
		model.cuda()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_ephoc = 0

	n = 1
	n_time = time.time()
	
	fout = open(log_filename, 'w')
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
					# print(outputs)
					# input()
					_, preds = torch.max(outputs, 1)
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
			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))
			
			fout.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format( phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_ephoc = epoch
				best_model_wts = copy.deepcopy(model.state_dict())
		

		torch.save(model, f'{dir_snapshot_out}/model_weights_epoch_{epoch}.pth')
		model_pruned_copy = torch.load(f'{dir_snapshot_out}/model_weights_epoch_{epoch}.pth')

		module_1 = model_pruned_copy.hl_1
		module_2 = model_pruned_copy.hl_2
		prune.remove(module_1, 'weight')
		prune.remove(module_2, 'weight')
		torch.save(model_pruned_copy, f'{dir_snapshot_out}/model_weights_epoch_{epoch}.pth')
		del model_pruned_copy
		
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
	torch.save(model, f'{dir_snapshot_out}/best_model.pth')
	return model

criterion = nn.CrossEntropyLoss()

# vgg16 = models.vgg16(pretrained=True)
# model = vgg.VGG(make_layers(cfgs['D'], batch_norm=False), num_classes=1000, init_weights=True)

# d_aux={
# 	'classifier.0.weight':'hl_1.weight',
# 	'classifier.0.bias':'hl_1.bias',
# 	'classifier.3.weight':'hl_2.weight',
# 	'classifier.3.bias':'hl_2.bias',
# 	'classifier.6.weight':'out_layer.weight',
# 	'classifier.6.bias':'out_layer.bias'}

# state_dict_pre_trained = vgg16.state_dict()
# state_dict = model.state_dict()
# for key in state_dict_pre_trained:
# 	#print (key)
# 	w_pre = state_dict_pre_trained[key].data.numpy()	
# 	if (key in state_dict):
# 		w_new = state_dict[key].data.numpy()
# 	else:
# 		w_new = state_dict[d_aux[key]].data.numpy()

# 	copy_bias(w_pre, w_new)


drop_db_dir	= '/media/revai/Data/Carlos/imagenet/50_classes/drop/'
val_db_dir	 = '/media/revai/Data/Carlos/imagenet/50_classes/test/'
map_label_file = '/media/revai/Data/Carlos/imagenet/map_index_label_50_classes.txt'

map_index_to_label = {}
f = open(map_label_file)
f = f.readlines()

for i in range(len(f)):
	line = f[i]
	map_index_to_label[line.split('\n')[0]] = i


X_drop = []
y_drop = []
X_val = []
y_val = []
for label in os.listdir(drop_db_dir):
	for img in os.listdir(drop_db_dir + '/' + label):
		y_drop.append(map_index_to_label[label])
		X_drop.append(drop_db_dir + '/' + label + '/' + img)
		
for label in os.listdir(val_db_dir):
	for img in os.listdir(val_db_dir + '/' + label):
		y_val.append(map_index_to_label[label])
		X_val.append(val_db_dir + '/' + label + '/' + img)



examples = [[-1,2], [2,1], [1,0], [2, -6], [1, 1], [-2, 2], [0,1], [5,-2]]
l_ace = {}
# model.cuda()
def is_act_neuron(x, mode = 'sigmoid', mean = None, std = None, threshold = None):
	if (mode == 'sigmoid'):
		if ((1.0 / (1 + math.exp(-x))) > 0.5): return 1
	elif (mode == 'mean_std'):
		if (((x - mean) / std) > threshold): return 1
	return 0

#l_threshold = [1, 2, 3, 4, 5]
l_threshold = [2]
N_HIDDEN_LAYERS = 2
num_neurons_hidden_layer = 4096
l_percent_to_drop = [0.9, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#l_percent_to_drop = [0.9, 0.95]
l_acc = {}
l_n_parameters = {}

preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def pruning_by_new_ace_approach(model, l_ace, percent=0.1, is_by_class=False, percent_classes_max_to_drop=0.05, is_by_both_class=False):
	#is_by_class = True
	drop_neurons = {}
	if (is_by_class):
		if (not is_by_both_class):
			#l_ace[label][l][i]
			max_classes_to_drop = len(l_ace) * percent_classes_max_to_drop
			max_classes_to_drop_dict = {}
			for label in l_ace:
				# print ('label: {}'.format(label))
				for n_layer in l_ace[label]:
					if (n_layer not in max_classes_to_drop_dict):
						max_classes_to_drop_dict[n_layer] = {}
					n_neurons_to_drop = int(len(l_ace[label][n_layer]) * percent)
					# print ('n_neurons_to_drop in layer {} = {}'.format(n_layer, n_neurons_to_drop))

					l_aux = []
					for idx_neuron in l_ace[label][n_layer]:
						if (n_neurons_to_drop == 0): break
						l_aux.append(idx_neuron)
						n_neurons_to_drop-=1
					# print (len(l_aux))
					if (n_layer not in drop_neurons):
						drop_neurons[n_layer] = []
						for i in l_aux:
							drop_neurons[n_layer].append(i)
					else:
						l_to_remove = []
						for n in drop_neurons[n_layer]:
							
							if (n not in l_aux):
								if (n not in max_classes_to_drop_dict[n_layer]):
									max_classes_to_drop_dict[n_layer][n] = 1
								max_classes_to_drop_dict[n_layer][n] += 1
								if (max_classes_to_drop_dict[n_layer][n] >= max_classes_to_drop):
									l_to_remove.append(n)
						for n in l_to_remove:
							drop_neurons[n_layer].remove(n)
					# print ('drop neuros: {}'.format(len(drop_neurons[n_layer])))
					# print ('------')
					# input()
		else:
			#l_ace[label][l][i]
			max_classes_to_drop = len(l_ace) * percent_classes_max_to_drop
			max_classes_to_drop_dict = {}

			for label in l_ace:
				for n_layer in l_ace[label]:
					max_classes_to_drop_dict[n_layer] = {}
					drop_neurons[n_layer] = []


			for label in l_ace:
				l_ace_aux = []
				for idx_neuron in l_ace[label][1]:
					l_ace_aux.append((l_ace[label][1][idx_neuron], idx_neuron, 1))
					l_ace_aux.append((l_ace[label][2][idx_neuron], idx_neuron, 2))
				l_ace_aux.sort()
				n_neurons_to_drop = int(len(l_ace_aux) * percent)
				l_aux = []
				for i in range(len(l_ace_aux)):
					if (n_neurons_to_drop == 0): break
					drop_neurons[l_ace_aux[i][2]].append(l_ace_aux[i][1])
					l_aux.append(l_ace_aux[i][1])
					n_neurons_to_drop-=1

				for n_layer in l_ace[label]:
					l_to_remove = []
					for n in drop_neurons[n_layer]:						
						if (n not in l_aux):
							if (n not in max_classes_to_drop_dict[n_layer]):
								max_classes_to_drop_dict[n_layer][n] = 1
							max_classes_to_drop_dict[n_layer][n] += 1
							if (max_classes_to_drop_dict[n_layer][n] >= max_classes_to_drop):
								l_to_remove.append(n)
					for n in l_to_remove:
						drop_neurons[n_layer].remove(n)
				# print ('drop neuros: {}'.format(len(drop_neurons[n_layer])))
				# print ('------')
				# input()

	else:
		if (not is_by_both_class):
			for n_layer in l_ace:
				drop_neurons[n_layer] = []
				n_neurons_to_drop = int(len(l_ace[n_layer]) * percent)
				for idx_neuron in l_ace[n_layer]:
					if (n_neurons_to_drop == 0): break
					#print ('{}: {}'.format(idx_neuron, l_ace[n_layer][idx_neuron]))
					drop_neurons[n_layer].append(idx_neuron)
					n_neurons_to_drop-=1

		#selecting neurons considering both layers in sort!
		else:
			l_ace_aux = []
			for idx_neuron in l_ace[1]:
				l_ace_aux.append((l_ace[1][idx_neuron], idx_neuron, 1))
				l_ace_aux.append((l_ace[2][idx_neuron], idx_neuron, 2))
			l_ace_aux.sort()

			n_neurons_to_drop = int(len(l_ace_aux) * percent)
			for n_layer in l_ace:
				drop_neurons[n_layer] = []

			for i in range(len(l_ace_aux)):
				if (n_neurons_to_drop == 0): break
				drop_neurons[l_ace_aux[i][2]].append(l_ace_aux[i][1])
				n_neurons_to_drop-=1
		# END --> selecting neurons considering both layers in sort!

	new_pruned_model = copy.deepcopy(model)

	# Build dependency graph
	DG = tp.DependencyGraph()
	DG.build_dependency(new_pruned_model, example_inputs=torch.randn(1,3,224,224))

	# get a pruning plan according to the dependency graph. idxs is the indices of pruned filters.
	pruning_plan = DG.get_pruning_plan( new_pruned_model.hl_1, tp.prune_linear, idxs=drop_neurons[1] )
	pruning_plan.exec()
	pruning_plan = DG.get_pruning_plan( new_pruned_model.hl_2, tp.prune_linear, idxs=drop_neurons[2] )
	pruning_plan.exec()

	return new_pruned_model

def compute_acc_from_model_(model, X_val, y_val, top_n=1):
	l_relu_values = {}
	l_acc   = [0.0 for i in range(top_n)]
	
	acc = 0.0
	count = 0
	
	start = time.time()
	for i in range(len(X_val)):
		filename = X_val[i]
		exp_out = y_val[i]
		#exp_out = np.argmax(exp_out)

		input_image = Image.open(filename)
		#print ('{}/{}: {}'.format(i, len(X_val), filename))
		#input_torchvar = autograd.Variable(torch.FloatTensor(inp), requires_grad=True)
		try:
			input_tensor = preprocess(input_image)
		except:
			#print("An exception occurred")
			continue
		count+=1
		input_batch = torch.unsqueeze(input_tensor, 0) # create a mini-batch as expected by the model

		# move the input and model to GPU for speed if available
		if torch.cuda.is_available():
			input_batch = input_batch.to(device)
			model.to(device)
		model.eval()
		out_model = model.forward_hidden_test(input_batch, 200)

		out_softmax = torch.nn.functional.softmax(out_model[-1], dim=-1)
		out_softmax = out_softmax.reshape(N_CLASSES)
		out_aux = {}
		i = 0
		for key in out_softmax.cpu().detach().numpy():
			out_aux[i] = key
			i+=1
		out_aux = dict( sorted(out_aux.items(), key=operator.itemgetter(1),reverse=True))
		
		count_ = 0
		for key in out_aux:
			if (count_ >= top_n): break

			if (exp_out == key):
				for j in range(count_, top_n):
					l_acc[j]+=1
				acc += 1
				break
			count_ +=1
	for j in range(top_n):
		l_acc[j] /= count
	end = time.time()
	print('Time do compute accuracy: {}'.format(end-start))
	return (acc/count), l_acc

#d_relu_mean = {1: 0.4717282, 2: 0.2339526}
#d_relu_std  = {1: 1.2522504, 2: 0.6938688}

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_ace_for_class(p_z, p_x_given_z, comb_z_x_index, p_y_given_x, label):
	ace = {1:{}, 2:{}}
	for x_index in range(4096):
		ace[1][x_index]=Decimal(0)
		ace[2][x_index]=Decimal(0)
	prod_p_z_comb = {}
	for layer in p_z:
		for z_index in p_z[layer]:
			total = float(sum(p_z[layer][z_index]))
			if (total != 0):
				p_z[layer][z_index][0] /= total
				p_z[layer][z_index][1] /= total

		for comb in comb_z_x_index[layer]:
			for x_index in comb_z_x_index[layer][comb]:
				total = float(sum(comb_z_x_index[layer][comb][x_index]))
				if (total != 0):
					comb_z_x_index[layer][comb][x_index][0]/=total
					comb_z_x_index[layer][comb][x_index][1]/=total

		for pa_y in p_y_given_x[layer]:
			total = float(sum(p_y_given_x[layer][pa_y]))
			if (total != 0):
				p_y_given_x[layer][pa_y][0]/=total
				p_y_given_x[layer][pa_y][1]/=total
		#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
		# for pa_y in p_y_given_x[layer][label]:
		# 	 total = float(sum(p_y_given_x[layer][label][pa_y]))
		# 	 if (total != 0):
		# 		 p_y_given_x[layer][label][pa_y][0]/=total
		# 		 p_y_given_x[layer][label][pa_y][1]/=total
		prod_p_z_comb[layer] = {}
		for comb_z in comb_z_x_index[layer]: #max 50
			p_z_ = Decimal(1)
			for z_index in range(len(comb_z)): #25k e 4k
				z_value = int(comb_z[z_index])
				v = Decimal(p_z[layer][z_index][z_value])
				p_z_ *= (v)
				if (p_z_ == 0):break
			prod_p_z_comb[layer][comb_z] = p_z_

	#print ('passssooouuu')

	for layer in p_x_given_z:
		print ('layer: {}. len = {}'.format(layer, len(p_x_given_z[layer])))
		for x_index in p_x_given_z[layer]: # 4096
			begin = timeit.default_timer()
			#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
			#for comb_x in p_y_given_x[layer][label]: #máx 50
			#	if (p_y_given_x[layer][label][comb_x][1] == 0): continue
			for comb_x in p_y_given_x[layer]: #max 50
				if (p_y_given_x[layer][comb_x][1] == 0): continue
				#só considera o x que ativado
				if (int(comb_x[x_index]) != 1): continue

				for comb_z in comb_z_x_index[layer]: #max 50
					p_z_ = Decimal(prod_p_z_comb[layer][comb_z])
					#for z_index in range(len(comb_z)): #25k e 4k
					#	z_value = int(comb_z[z_index])
					#	v = Decimal(p_z[layer][z_index][z_value])
					#	p_z_ *= (v)
					#	if (p_z_ == 0):break					
					if (p_z_ == 0): continue

					p_x = Decimal(1)
					for x_i in range(len(comb_x)): #50
						#desconsidera o x que estamos computando o ACE
						if (x_i == x_index): continue

						x_value = int(comb_x[x_i])
						p_x *= Decimal(comb_z_x_index[layer][comb_z][x_i][x_value])
						if (p_x == 0): break
					if (p_x == 0): continue
					#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
					#ace[layer][x_index] += p_z_*p_x*Decimal(p_y_given_x[layer][label][comb_x][1])
					ace[layer][x_index] += p_z_*p_x*Decimal(p_y_given_x[layer][comb_x][1])
			end = timeit.default_timer()
			#print ('{}/{} ({}s) = {}'.format(x_index+1, len(p_x_given_z[layer]), end - begin, ace[layer][x_index]))

		#ace[layer] = dict(sorted(ace[layer].items(), key=operator.itemgetter(1),reverse=False))
		#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
		ace[layer] = dict(sorted(ace[layer].items(), key=operator.itemgetter(1),reverse=False))
		# for x in ace[layer]:
		#	 print ('{}: {}'.format(x, ace[layer][x]))
		#	 input()
	return ace

def init_variables():
	p_z = {1:{}, 2:{}}
	p_x_given_z = {1:{}, 2:{}}
	comb_z_x_index = {1:{}, 2:{}}
	p_y_given_x = {1:{}, 2:{}}
	#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
	#for i in range(1000):
	#	p_y_given_x[1][i] = {}
	#	p_y_given_x[2][i] = {}
	for z_index in range(25088):
		p_z[1][z_index] = [0, 0]
	for z_index in range(4096):
		p_z[2][z_index] = [0, 0]
		p_x_given_z[1][z_index]={}
		p_x_given_z[2][z_index]={}
	return p_z, p_x_given_z, comb_z_x_index, p_y_given_x
import random
random.seed(10081993)
def compute_ace(X_drop, y_drop, is_by_both_class):
	global prefix_filename
	global d_relu_mean 
	global d_relu_std   
	#fout = open(prefix_filename + '.csv', 'w')
	for threshold in l_threshold:
		#l_ace = {1:random.sample(range(0, 4096), 4096),
		#		 2:random.sample(range(0, 4096), 4096)}
		if (True):
			p_z, p_x_given_z, comb_z_x_index, p_y_given_x = init_variables()
			label = y_drop[0]
			#l_ace[label] = {}
			start_proc_imgs = timeit.default_timer()
			for i in range(len(X_drop)):
				if (i%1000 == 0):
					stop_aux = timeit.default_timer()
					print ('processing img #{} ({}s)'.format(i, stop_aux - start_proc_imgs))
					start_proc_imgs = timeit.default_timer()
				filename = X_drop[i]
				exp_out = y_drop[i]
				
				#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
				#if (exp_out != label):
				#	 l_ace[label] = compute_ace_for_class(p_z, p_x_given_z, comb_z_x_index, p_y_given_x, label)
				#	 p_z, p_x_given_z, comb_z_x_index, p_y_given_x = init_variables()
				#	 label = exp_out

				input_image = Image.open(filename)
				input_tensor  = None
				try: input_tensor = preprocess(input_image)
				except: continue
				input_batch = torch.unsqueeze(input_tensor, 0) # create a mini-batch as expected by the model
				
				# move the input and model to GPU for speed if available
				#if torch.cuda.is_available(): 
				input_batch = input_batch.to(device)
				model.eval()
				with torch.no_grad():
					out_model = model.forward(input_batch)

				out_softmax = torch.nn.functional.softmax(out_model[-1], dim=-1)
				out_softmax = out_softmax.reshape(N_CLASSES)
				
				out_softmax = np.argmax(out_softmax.cpu().detach().numpy())
				
				for layer in range(1, N_HIDDEN_LAYERS + 1):
					mean = d_relu_mean[layer]
					std = d_relu_std[layer]
					out = out_model[layer].cpu().detach().numpy()
					comb = ''
					out_previous_layer = out_model[layer-1].cpu().detach().numpy()
					
					for z_index in range(len(out_previous_layer[0])):
						is_act_z = is_act_neuron(x = out_previous_layer[0][z_index], mode = 'mean_std', mean=mean, std=std, threshold=threshold)
						p_z[layer][z_index][is_act_z]+=1
						comb += str(is_act_z)
					if (comb not in comb_z_x_index[layer]): comb_z_x_index[layer][comb]={}
					comb_pa_y = ''
					for x_index in range(len(out[0])):
						if (x_index not in comb_z_x_index[layer][comb]): comb_z_x_index[layer][comb][x_index] = [0, 0]
						if (comb not in p_x_given_z[layer][x_index]): p_x_given_z[layer][x_index][comb] = [0, 0]
						
						x_value = out[0][x_index]
						is_act_x = is_act_neuron(x_value, mode = 'mean_std', mean=mean, std=std, threshold=threshold) 
						comb_pa_y += str(is_act_x)
						p_x_given_z[layer][x_index][comb][is_act_x]+=1
						comb_z_x_index[layer][comb][x_index][is_act_x]+=1
					
					if (comb_pa_y not in p_y_given_x[layer]):
						p_y_given_x[layer][comb_pa_y] = [0, 0]

					if (out_softmax == exp_out):
						p_y_given_x[layer][comb_pa_y][1]+=1
					else:
						p_y_given_x[layer][comb_pa_y][0]+=1
					#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
					# if (comb_pa_y not in p_y_given_x[layer][out_softmax]):
					# 	 p_y_given_x[layer][out_softmax][comb_pa_y] = [0, 0]

					# if (out_softmax == exp_out):
					# 	 p_y_given_x[layer][out_softmax][comb_pa_y][1]+=1
					# else:
					# 	 p_y_given_x[layer][out_softmax][comb_pa_y][0]+=1
			#NA TENTATIVA DE COMPUTAR O ACE POR CLASSE!!!!
			#l_ace[label] = compute_ace_for_class(p_z, p_x_given_z, comb_z_x_index, p_y_given_x, label)
			l_ace = compute_ace_for_class(p_z, p_x_given_z, comb_z_x_index, p_y_given_x, label)

			fout_ace = open(prefix_filename + '/ace.txt','w')
			for layer in l_ace:
				fout_ace.write('layer {}\n'.format(layer))
				fout_ace.write('{}\n'.format(l_ace[layer]))

		for percent in l_percent_to_drop:
			start_pruning = timeit.default_timer()
			print ('percent to pruning for threshold = {}: {}'.format( percent, threshold))
			#model_clone = copy.deepcopy(model)
			#for percent_classes_max_to_drop in [0, 0.05, 0.1, 0.2, 0.3]:
			pruned_model = pruning_by_new_ace_approach(model, l_ace, percent, is_by_class=False, percent_classes_max_to_drop=0.05, is_by_both_class=is_by_both_class)

			# Observe that all parameters are being optimized
			optimizer_ft = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
			# Decay LR by a factor of 0.1 every 7 epochs
			exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
			dir_snapshot_out = f'{prefix_filename}/snapshots_{percent}/'

			if not os.path.exists(dir_snapshot_out): os.makedirs(dir_snapshot_out)
			log_filename = f'{prefix_filename}/log_{percent}.txt'
			torch.save(pruned_model, f'{dir_snapshot_out}/prunned_model_ori.pth')

			train_model(pruned_model, criterion=criterion,optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
			num_epochs=25, log_filename=log_filename, dir_snapshot_out=dir_snapshot_out)
			#torch.save(pruned_model, f'{dir_snapshot_out}/pruned_model_model_weights.pth')

			# acc = compute_acc_from_model_(pruned_model, X_val, y_val, top_n=5)
			# #l_acc[threshold][percent] = acc
			# n_parameters = count_parameters(pruned_model)
			del pruned_model
			torch.cuda.empty_cache()
			# stop_pruning = timeit.default_timer()
			# print('Time to prunning and compute acc for new model: ', stop_pruning - start_pruning)
			# for i in range(len(acc[1])):
			# 	acc[1][i] = str(acc[1][i]).replace('.', ',')
			# fout.write('{};{};{};{};{};{};{};{};\n'.format(threshold, percent ,n_parameters, acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4]))
			# print ('{};{};{};{};{};{};{};{};\n'.format(threshold, percent ,n_parameters, acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4]))
			# print ('acc = {}'.format(acc[1][0]))
			# print ('n_parameters = {}'.format(n_parameters))
			# print ('-----')


def compute_mean_std(X_drop, y_drop, model):
	d_relu_values = {}

	for i in range(len(X_drop)):
		if (i%1000 == 0): print ('processing img #{}'.format(i))
		filename = X_drop[i]
		exp_out = y_drop[i]

		input_image = Image.open(filename)
		try:
			input_tensor = preprocess(input_image)
		except:
			continue

		input_tensor = preprocess(input_image)
		input_batch = torch.unsqueeze(input_tensor, 0) # create a mini-batch as expected by the model

		# move the input and model to GPU for speed if available
		if torch.cuda.is_available():
			input_batch = input_batch.to(device)
			model.to(device)

		model.eval()
		with torch.no_grad():
			out_model = model.forward(input_batch)
		#out_model = model.forward(input_batch)

		out_softmax = torch.nn.functional.softmax(out_model[-1], dim=-1)
		out_softmax = out_softmax.reshape(N_CLASSES)

		out_softmax = np.argmax(out_softmax.cpu().detach().numpy())
		#print (len(out_model))
		#input()
		for j in range(1, len(out_model)-1):
			if (j not in d_relu_values): d_relu_values[j] = []

			out = out_model[j].cpu().detach().numpy()

			out_previous_layer = out_model[j-1].cpu().detach().numpy()

			for k in range(len(out[0])):
				v = out[0][k]
				d_relu_values[j].append(v)

	for l in d_relu_values:
		d_relu_values[l] = list(d_relu_values[l])

	d_relu_mean = {}
	d_relu_std  = {}

	for l in d_relu_values:
		d_relu_mean[l] = np.mean(d_relu_values[l])
		d_relu_std[l]  = np.std(d_relu_values[l])

	return d_relu_mean, d_relu_std

def runing_by_pytorch_module(model, technique='ln_structured', amount_=0.3, is_global_pruning=False):

	module_1 = model.hl_1
	module_2 = model.hl_2


	if (not is_global_pruning):
		if (technique == 'ln_structured'):
			print ('#### ln_structured ####')
			prune.ln_structured(module_1, name="weight", amount=amount_, n=1, dim=1)
			prune.ln_structured(module_2, name="weight", amount=amount_, n=1, dim=1)

		elif (technique=='l1_unstructured'):
			print ('#### l1_unstructured ####')
			prune.l1_unstructured(module_1, name="weight", amount=amount_)
			prune.l1_unstructured(module_2, name="weight", amount=amount_)
		
		elif (technique=='random_structured'):
			print ('#### random_structured ####')
			prune.random_structured(module_1, name="weight", amount=amount_, dim=1)
			prune.random_structured(module_2, name="weight", amount=amount_, dim=1)

		elif (technique == 'random_unstructured'):
			print ('#### random_unstructured ####')
			prune.random_unstructured(module_1, name="weight", amount=amount_)
			prune.random_unstructured(module_2, name="weight", amount=amount_)

		else:
			print ('#### ln_structured ####')
			prune.ln_structured(module_1, name="weight", amount=amount_, n=1, dim=1)
			prune.ln_structured(module_2, name="weight", amount=amount_, n=1, dim=1)

	else:
		if (technique == 'l1_unstructured'):
			print ('#### global_unstructured l1_unstructured####')
			parameters_to_prune = [(module_1, "weight"), (module_2, "weight")]
			prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount_)


	columns_pruned = int(sum(torch.sum(module_1.weight, dim=0) == 0))
	columns_pruned += int(sum(torch.sum(module_2.weight, dim=0) == 0))
	print ('#pruned_parameters: {}'.format(columns_pruned))

	#prune.remove(module_1, 'weight')
	#prune.remove(module_2, 'weight')


	return None

def measure_sparsity(module, weight=True, bias=False):
	num_zeros = 0
	num_elements = 0
	for param_name, param in module.named_parameters():
		#print (param_name)
		if "weight" in param_name and weight == True:
			num_zeros += torch.sum(param == 0).item()
			num_elements += param.nelement()
		if "bias" in param_name and bias == True:
			num_zeros += torch.sum(param == 0).item()
			num_elements += param.nelement()
	print ('sparsity: {}'.format(float(num_zeros)/num_elements))
	return num_zeros

# max_acc = -1
# max_acc_model_path = ''
# f_out_acc = open('acc_frozen_layer.txt', 'w')
# for filemodel in os.listdir('/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/'):
# 	print (f'/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/{filemodel}')
# 	model_test = torch.load(f'/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/{filemodel}')
# 	acc, l_acc = compute_acc_from_model_(model_test, X_val, y_val, top_n=5)
# 	f_out_acc.write(f'{filemodel}: {l_acc}\n')
# 	if (l_acc[-1] > max_acc):
# 		max_acc = l_acc[-1]
# 		max_acc_model_path = f'/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/{filemodel}'

# f_out_acc.close()
# print (f'best snapshot: {map_index_to_label}, acc: {max_acc}')
# print ('DONE acc')
# input()

# ######################################
import copy
model_test = vgg.VGG(make_layers(cfgs['D'], batch_norm=False), num_classes=1000, init_weights=True)

dataset__dir_test = '/media/revai/Data/Carlos/ace_new_carlos_imp/results/exp_finnetuning_freezed_layers/'
for subdir in os.listdir(f'{dataset__dir_test}'):
	if (subdir not in ['l1_unstructured', 'random_structured', 'random_unstructured']): continue
	f_out = open(f'res_finnetuning_{subdir}.txt', 'w') 
	for t in os.listdir(f'{dataset__dir_test}/{subdir}'):
		if ('snapshots' not in t): continue
		max_acc = -1
		max_l_acc = []
		best_model = None
		filename_max = ''
		for m in os.listdir(f'{dataset__dir_test}/{subdir}/{t}'):
			if ('.pth' not in m): continue
			model_test = torch.load(f'{dataset__dir_test}/{subdir}/{t}/{m}')

			model_test.eval()
			acc, l_acc = compute_acc_from_model_(model_test, X_val, y_val, top_n=5)
			sss = str(l_acc).replace('[', '').replace(']', '').replace(',',';').replace(' ', '')
			f_out.write(f'{subdir};{t};{m};{sss};\n')
			if (acc > max_acc):
				max_acc = acc
				max_l_acc = l_acc.copy()
				#best_model = copy.deepcopy(model_test)
				filename_max = f'{dataset__dir_test}/{subdir}/{t}/{m}'
			#print ('{}: {}'.format(m, acc))

		print ('{}: {}'.format(filename_max, max_l_acc))
		#acc = compute_acc_from_model_(model, X_val, y_val, top_n=5)
		#print('acc pre-trained model')
		#print (acc)
		#torch.save(best_model, f'{dataset__dir_test}/{subdir}/{t}/best_of_the_best_model.pth')
	f_out.close()
print('DONE')
input()
######################################
model_ = torch.load('/media/revai/Data/Carlos/ace_new_carlos_imp/snapshots_50_sgd_freezed_conv_layers/model_weights_epoch_92.pth')
model = copy.deepcopy(model_)
#input()
# prefix_filename = './results/exp_finnetuning_freezed_layers/random/'
# if not os.path.exists(prefix_filename): os.makedirs(prefix_filename)
# is_by_both_class=False

# # d_relu_mean, d_relu_std = compute_mean_std(X_drop, y_drop, model)
# compute_ace(X_drop, y_drop, is_by_both_class)
# print ('DONE RANDOM PRUNING')
# exit()
# input()
model = copy.deepcopy(model_)
l_techniques = ['l1_unstructured', 'random_structured', 'random_unstructured']
#l_techniques = ['ln_structured']
#l_techniques = ['l1_unstructured']
for technique in l_techniques:
	prefix_filename = f'./results/exp_finnetuning_freezed_layers/{technique}'
	if not os.path.exists(prefix_filename): os.makedirs(prefix_filename)
	#prefix_filename = './results/exp_08_11_2021/{}'.format(technique)
	#fout = open(prefix_filename + '/res.csv', 'w')
	for percent in l_percent_to_drop:
		start_pruning = timeit.default_timer()
		print ('percent to prune with {} technique = {}'.format(technique, percent))
		#model_clone = copy.deepcopy(model)
		#for percent_classes_max_to_drop in [0, 0.05, 0.1, 0.2, 0.3]:
		pruned_model = copy.deepcopy(model)

		n_param = measure_sparsity(pruned_model.hl_1) + measure_sparsity(pruned_model.hl_2)

		pruning_by_pytorch_module(pruned_model, technique=technique, amount_=percent, is_global_pruning=False)

		# Observe that all parameters are being optimized
		optimizer_ft = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
		# Decay LR by a factor of 0.1 every 7 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
		dir_snapshot_out = f'{prefix_filename}/snapshots_{percent}/'

		if not os.path.exists(dir_snapshot_out): os.makedirs(dir_snapshot_out)
		log_filename = f'{prefix_filename}/log_{percent}.txt' 

		torch.save(pruned_model, f'{dir_snapshot_out}/prunned_model_ori.pth')
		model_pruned_copy = torch.load(f'{dir_snapshot_out}/prunned_model_ori.pth')

		module_1 = model_pruned_copy.hl_1
		module_2 = model_pruned_copy.hl_2
		prune.remove(module_1, 'weight')
		prune.remove(module_2, 'weight')

		torch.save(model_pruned_copy, f'{dir_snapshot_out}/prunned_model_ori.pth')
		print (f'num zeros: {measure_sparsity(model_pruned_copy)}')
		del model_pruned_copy
		train_model(pruned_model, criterion=criterion,optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
		num_epochs=25, log_filename=log_filename, dir_snapshot_out=dir_snapshot_out)

		# acc = compute_acc_from_model_(pruned_model, X_val, y_val, top_n=5)
		# #l_acc[threshold][percent] = acc
		# n_parameters = count_parameters(pruned_model) - n_param
		del pruned_model
		torch.cuda.empty_cache()
		stop_pruning = timeit.default_timer()
		# print('Time to prunning and compute acc for new model: ', stop_pruning - start_pruning)
		# for i in range(len(acc[1])):
		# 	acc[1][i] = str(acc[1][i]).replace('.', ',')
		# out_aux = '{};{};{};{};{};{};{};\n'.format(percent ,n_parameters, acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4])
		# out_aux.replace('.', ',')
		# fout.write(out_aux)
		# print (out_aux)
		# print ('acc = {}'.format(acc[1][0]))
		# print ('n_parameters = {}'.format(n_parameters))
		# print ('-----')
	print ('#############')
	print ('#############')
	print ('#############')
	#fout.close()

#print ('d_relu_mean: {}'.format(d_relu_mean))
#print ('d_relu_std: {}'.format(d_relu_std))



'''
d_relu_mean= {1: 0.5341921, 2: 0.25907692}
d_relu_std= {1: 1.3999276, 2: 0.7812767}
'''

