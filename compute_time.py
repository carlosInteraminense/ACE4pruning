from torchvision import transforms
import time
import torch
import pandas as pd
import os
import glob
from PIL import Image
import warnings
import torch.nn.utils.prune as prune
warnings.simplefilter("ignore")

preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

snapshot_dir = '/media/revai/Data/Carlos/ace_new_carlos_imp/results/exp_finnetuning_freezed_layers/'
filename_model = 'best_model.pth'

l_percent_to_drop = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
l_approach = ['ace', 'l1_unstructured', 'random']

# for prune_percent in l_percent_to_drop:
#     d_models = {}
#     for approach in l_approach:
#         model_path = f'{snapshot_dir}/{approach}/snapshots_{prune_percent}/{filename_model}'
#         model = torch.load(model_path)

filename_best_model = '02_best_model.pth'
l_models = glob.glob(f'{snapshot_dir}/ace/**/{filename_best_model}', recursive=True)
l_models.extend(glob.glob(f'{snapshot_dir}/ln_structured/**/{filename_best_model}', recursive=True))
l_models.extend(glob.glob(f'{snapshot_dir}/random/**/{filename_best_model}', recursive=True))
l_models.extend(glob.glob(f'{snapshot_dir}/original_model/**/{filename_best_model}', recursive=True))

#device = 'cpu'
test_db_dir	 = '/media/revai/Data/Carlos/imagenet/50_classes/test/'

x_test = []		
for label in os.listdir(test_db_dir):
    count = 0    
    for img in os.listdir(test_db_dir + '/' + label):
        if (count == 10): break
        x_test.append(test_db_dir + '/' + label + '/' + img)
        count+=1

count = 0
df = pd.DataFrame({})
for __i in range (10):
    start_round = time.time()
    print (f'round: {__i}...')
    for model_path in l_models:
        #if ('ln_structured' not in model_path): continue
        # model_pruned_copy = torch.load(model_path)

        # module_1 = model_pruned_copy.hl_1
        # module_2 = model_pruned_copy.hl_2
        # prune.remove(module_1, 'weight')
        # prune.remove(module_2, 'weight')
        # ff = model_path.split('/')[-1]
        # model_out_path = model_path.replace(ff, f'02_{ff}')
        # torch.save(model_pruned_copy, model_out_path)
        # continue
        model = torch.load(model_path, map_location=torch.device('cpu'))
        for i in range(len(x_test)):
            filename = x_test[i]

            input_image = Image.open(filename)
            try: input_tensor = preprocess(input_image)
            except: continue
            count+=1
            input_batch = torch.unsqueeze(input_tensor, 0) # create a mini-batch as expected by the model

            model.eval()
            start = time.time()
            out_model = model.forward(input_batch)
            end = time.time()
            exec_time = (end - start) * 1000

            pruned_percent = model_path.split('/')[-2]
            prune_technique = model_path.split('/')[-3]

            df = pd.concat([df, pd.DataFrame.from_records([{'prune_technique':prune_technique, 
                        'pruned_percent':pruned_percent,
                        'round': __i+1,
                        'filename': filename,
                        'time_ms': exec_time}])], ignore_index=True)
        df.to_csv('results/time_result_13_03_23.csv')
    
    end_round = time.time()
    print (f'total time: {end_round - start_round} seconds')
    time.sleep(300)