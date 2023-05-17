fin = open('res_finnetuning_random.txt')

l_lines = fin.readlines()

best_results = {
    'ace': {},
    'ln_structured': {},
    'random': {}
}

prunned_results = {
    'ace': {},
    'ln_structured': {},
    'random': {}
}


for line in l_lines:
    line = line.split('\n')[0].split(';')
    print (line)
    approach = line[0]
    prune_percent = line[1].split('_')[-1]
    l_acc = line[3:-1]
    if ('prunned_model_ori' in line[2]):
        if (prune_percent  not in prunned_results[approach]):
            prunned_results[approach][prune_percent] = l_acc
            prunned_results[approach]['filename_model'] = line[2]
        else:
            l_acc_ = prunned_results[approach][prune_percent]
            if (float(line[-2]) > float(l_acc_[-1])):
                prunned_results[approach][prune_percent] = l_acc_
                prunned_results[approach]['filename_model'] = line[2]
            

    else:
        if (prune_percent  not in best_results[approach]):
            best_results[approach][prune_percent] = l_acc
            best_results[approach]['filename_model'] = line[2]
        else:
            l_acc_ = best_results[approach][prune_percent]
            #print ('OOOOOOOOOOOOOOOOIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
            if (float(line[-2]) > float(l_acc_[-1])):
                best_results[approach][prune_percent] = l_acc_
                best_results[approach]['filename_model'] = line[2]


#result =   open('res_finnetuning_random_best_result_finetuning.csv', 'w')
#result_2 = open('res_finnetuning_random_best_result_not_finetuning.csv', 'w')

#result.write(f'approach;prune_percent;filename_model;top1;top2;top3;top4;top5\n')
#result_2.write(f'approach;prune_percent;filename_model;top1;top2;top3;top4;top5\n')

for approach in best_results:
    for prune_percent in best_results[approach]:
        if (prune_percent == 'filename_model'): continue
        #print (f'{approach};{prune_percent};{prunned_results[approach]["filename_model"]};')
        print (f'{approach};{prune_percent};{best_results[approach]["filename_model"]};')
        #result.write(f'{approach};{prune_percent};{best_results[approach]["filename_model"]};')
        #result_2.write(f'{approach};{prune_percent};{prunned_results[approach]["filename_model"]};')
        l_acc = best_results[approach][prune_percent]
        #result.write(f'{";".join([x for x in l_acc])}\n'.replace(',', '.'))
        l_acc = prunned_results[approach][prune_percent]
        #result_2.write(f';{";".join([x for x in l_acc])}\n'.replace(',', '.'))

#result.close()
#result_2.close()