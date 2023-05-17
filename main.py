import torch
import math
import my_neural_network

model = my_neural_network.NN()

examples = [[-1,2], [2,1], [1,0], [2, -6], [1, 1], [-2, 2], [0,1], [5,-2]]
l_ace = {}

def is_act_neuron(x):
    if ((1.0 / (1 + math.exp(-x))) > 0.5): return 1
    return 0

def compute_ace_for_x(p_z, p_y_given_x, comb_z_x_index):
    global l_ace
    for x_index in l_ace:
        for comb_x in p_y_given_x:
            if (p_y_given_x[comb_x][1] == 0): continue

            #só considera o x que ativado
            if (int(comb_x[x_index]) != 1): continue
            for comb_z in comb_z_x_index:
                p_z_ = 1
                for z_index in range(len(comb_z)):
                    z_value = int(comb_z[z_index])
                    p_z_ *= (p_z[z_index][z_value])

                p_x=1
                for x_i in range(len(comb_x)):
                    #desconsidera o x que estamos computando o ACE
                    if (x_i == x_index): continue

                    x_value = int(comb_x[x_i])

                    p_x *= comb_z_x_index[comb_z][x_i][x_value]
                l_ace[x_index] += p_z_*p_x*p_y_given_x[comb_x][1]
    return l_ace

p_z = {}
p_x_given_z = {}
comb_z_x_index = {}
p_y_given_x = {} #se tiver mais de uma saída (multiclass) tem que ajustar aqui..

for z_index in range(len(examples[0])):
    p_z[z_index] = [0, 0]

p = torch.tensor(examples[0])
p = p.float()
out_model = model.forward(p)
neurons_values = out_model[0].cpu().detach().numpy()

for x_index in range(len(neurons_values)):
    p_x_given_z[x_index] = {}
    l_ace[x_index] = 0


#compute_act_neurons
for i in examples:
    p = torch.tensor(i)
    p = p.float()
    out_model = model.forward(p)
    layer = 0
    neurons_values = out_model[layer].cpu().detach().numpy()
    pred = out_model[-1].cpu().detach().numpy()

    previous_layer_neurons_values = i

    comb = ''
    for z_index in range(len(previous_layer_neurons_values)):
        z = previous_layer_neurons_values[z_index]
        is_act_z = is_act_neuron(z)
        p_z[z_index][is_act_z]+=1
        comb += str(is_act_z)
    if (comb not in comb_z_x_index): comb_z_x_index[comb]={}
    #provavelmente tem que iterar pelas camadas aqui e na ultima camada
    # computar a comb dos pais de y..
    comb_pa_y = ''
    for x_index in range(len(neurons_values)):
        if (x_index not in comb_z_x_index[comb]): comb_z_x_index[comb][x_index] = [0, 0]
        if (comb not in p_x_given_z[x_index]): p_x_given_z[x_index][comb] = [0, 0]
        
        x_value = neurons_values[x_index]
        is_act_x = is_act_neuron(x_value)
        comb_pa_y += str(is_act_x)
        p_x_given_z[x_index][comb][is_act_x]+=1
        comb_z_x_index[comb][x_index][is_act_x]+=1   
    
    if (comb_pa_y not in p_y_given_x):
        p_y_given_x[comb_pa_y] = [0, 0]
    
    is_act_pred = is_act_neuron(pred)
    p_y_given_x[comb_pa_y][is_act_pred]+=1

# compute probabilities
for z_index in p_z:
    total = float(sum(p_z[z_index]))
    p_z[z_index][0] /= total
    p_z[z_index][1] /= total

comb_z_x_index[comb][x_index] = [0, 0]
for comb in comb_z_x_index:
    for x_index in comb_z_x_index[comb]:
        total = float(sum(comb_z_x_index[comb][x_index]))
        if (total != 0):
            comb_z_x_index[comb][x_index][0]/=total
            comb_z_x_index[comb][x_index][1]/=total

for x_index in p_x_given_z:
    for comb in p_x_given_z[x_index]:
        total = float(sum(p_x_given_z[x_index][comb]))
        p_x_given_z[x_index][comb][0]/=total
        p_x_given_z[x_index][comb][1]/=total

for pa_y in p_y_given_x:
    total = float(sum(p_y_given_x[pa_y]))
    p_y_given_x[pa_y][0]/=total
    p_y_given_x[pa_y][1]/=total

l_ace = compute_ace_for_x(p_z, p_y_given_x, comb_z_x_index)
print (l_ace)