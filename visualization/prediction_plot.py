"""
Created on 6/1/19

@author: Baoxiong Jia

Description:

"""
import os
import re
import numpy as np
import seaborn as sns
rc={'axes.labelsize': 20, 'font.size': 20, 'legend.fontsize': 20.0, 'axes.titlesize': 20, 'xtick.labelsize': 20.0, 'ytick.labelsize': 24.0,}
sns.set(rc=rc)
import pandas as pd
import matplotlib.pyplot as plt

path = '/media/hdd/home/baoxiong/Projects/TPAMI2019/tmp/cad/log/nn_results'
save_path = '/media/hdd/home/baoxiong/Projects/TPAMI2019/fig'
pred_duration = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
gep_paths = [os.path.join(path, 'gep_pred{}_eval.txt'.format(i)) for i in pred_duration]
nn_paths = [os.path.join(path, 'pred{}_eval.txt'.format(i)) for i in pred_duration]

pattern = '[0-9]+.[0-9]+'

df_columns = ['Prediction duration (s)', 'LSTM + GEP', 'LSTM', 'Random']
df = []
for idx, paths in enumerate(zip(gep_paths, nn_paths)):
    gep_path, nn_path = paths
    with open(gep_path, 'r') as f:
        results_gep = f.readlines()
        gep_acc = float(re.findall(pattern, results_gep[-1])[-1])
        # gep_acc = 0
        print('gep acc', gep_acc)
    with open(nn_path, 'r') as f:
        results_nn = f.readlines()
        nn_acc = float(re.findall(pattern, results_nn[-1])[-1])
        print('nn acc', nn_acc)
    df.append([pred_duration[idx] / 15, gep_acc, nn_acc, 0.1])
df = pd.DataFrame(df, columns=df_columns)
fig, ax = plt.subplots()
plt.axes([0, 0, 1 / 0.618, 1])
df = pd.melt(df, id_vars=df_columns[0], value_vars=df_columns[1 : ], var_name='Method', value_name='F1 score')
ax = sns.lineplot(x=df_columns[0], y='F1 score', hue='Method', data=df)
ax.lines[2].set_linestyle('--')
ax.set_title(r'Frame prediction over time')
ax.set(xticks = np.array(pred_duration) / 15)
plt.ylim(0, 0.7)
ax.legend(loc='upper right')
print(sns.plotting_context())
plt.savefig(os.path.join(save_path, 'cad_prediction.pdf'), bbox_inches='tight')
