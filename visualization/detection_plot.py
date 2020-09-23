"""
Created on 6/1/19

@author: Baoxiong Jia

Description:

"""

import re
import os
import glob
import seaborn as sns
rc={'axes.labelsize': 28, 'font.size': 20, 'legend.fontsize': 20.0, 'axes.titlesize': 20, 'xtick.labelsize': 24.0, 'ytick.labelsize': 28.0,}
sns.set(rc=rc)
import pandas as pd
import matplotlib.pyplot as plt

path = '/media/hdd/home/baoxiong/Projects/TPAMI2019/tmp/breakfast/log/'

save_path = '/media/hdd/home/baoxiong/Projects/TPAMI2019/fig'
subsample_rate = [1, 2, 5, 10, 20, 50]
# trained_epochs = [5, 10, 15, 20, 25, 30, 35, 40]
trained_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

pattern = '[0-9]+.[0-9]+'

gep_all_paths = [[glob.glob(os.path.join(path, 'gep_results', 'eval_s{}_*_t{}.txt'.format(s, t)))[0] for t in trained_epochs] for s in subsample_rate]
nn_all_paths = [[glob.glob(os.path.join(path, 'nn_results', 'eval_s{}_*_t{}.txt'.format(s, t)))[0] for t in trained_epochs] for s in subsample_rate]

df_columns = ['Trained epochs', 'Bi-LSTM + GEP', 'Bi-LSTM']
for s_idx, (gep_paths, nn_paths) in enumerate(zip(gep_all_paths, nn_all_paths)):
    df = []
    for t_idx, (gep_path, nn_path) in enumerate(zip(gep_paths, nn_paths)):
        with open(gep_path, 'r') as f:
            results_gep = f.readlines()
            gep_acc = float(re.findall(pattern, results_gep[-3])[0])
        with open(nn_path, 'r') as f:
            results_nn = f.readlines()
            nn_acc = float(re.findall(pattern, results_nn[-1])[0])
        print(gep_acc, nn_acc)
        df.append([trained_epochs[t_idx], gep_acc, nn_acc])
    df = pd.DataFrame(df, columns=df_columns)
    fig, ax = plt.subplots()
    df = pd.melt(df, id_vars=df_columns[0], value_vars=df_columns[1 :], var_name='Method', value_name='Accuracy')
    sns.lineplot(x=df_columns[0], y='Accuracy', hue='Method', data=df)
    plt.xticks(trained_epochs)
    plt.ylim(0, 0.7)
    # plt.title(r'Detection result with {} frame subsample'.format(subsample_rate[s_idx]) if subsample_rate[s_idx] != 1
    #           else r'Detection result w/o subsample')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, 'breakfast_subsample_{}.pdf'.format(subsample_rate[s_idx])), bbox_inches='tight')
    print('Finished for {}'.format(subsample_rate[s_idx]))

