import os
import shutil

d_n_images = {
    'val':50,
    'test':50,
    'drop':50,
    'train':-1
}

l_dataset_dir = ['/media/revai/Data/Carlos/imagenet/imagenet/val/', '/media/revai/Data/Carlos/imagenet/imagenet/train/']
out_dir = '/media/revai/Data/Carlos/imagenet/splited_/'

d_label_filename = {}

for directory in l_dataset_dir:
    for label in os.listdir(directory):
        if label not in d_label_filename: d_label_filename[label] = [ directory + '/' + label + '/' + filename for filename in os.listdir(directory + '/' + label + '/')]
        else: d_label_filename[label].extend([ directory + '/' + label + '/' + filename for filename in os.listdir(directory + '/' + label + '/')])

os.makedirs(out_dir + '/val/')
os.makedirs(out_dir + '/test/')
os.makedirs(out_dir + '/drop/')
os.makedirs(out_dir + '/train/')

for label in d_label_filename:
    n_imgs = len(d_label_filename[label])
    index = 0
    for sub_dir in d_n_images:
        os.mkdir('{}/{}/{}/'.format(out_dir, sub_dir, label))
        n_imgs_for_sub_dir = d_n_images[sub_dir]
        if (n_imgs_for_sub_dir ==-1):n_imgs_for_sub_dir=n_imgs-150
        for i in range(index, index + n_imgs_for_sub_dir):
            shutil.copy(d_label_filename[label][i], '{}/{}/{}/'.format(out_dir, sub_dir, label))
            index+=1
        
