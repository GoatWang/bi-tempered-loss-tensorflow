import os
import re
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
temp_dir = os.path.join('results')
accuracy_last_dir = os.path.join('results', 'accuracy_last')
accuracy_max_dir = os.path.join('results', 'accuracy_max')
for d in [temp_dir, accuracy_last_dir, accuracy_max_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

with open('TrainingLog.txt', 'r', encoding='utf8') as f:
    rows = []
    starts = False
    for line in f:
        if "t1" in line:
            starts = True
            row = {}
            accuracy_train = []
            accuracy_test = []

        if starts:
            if 't1' in line:
                row['t1'] = float(line.split(': ')[1])
            elif 't2' in line:
                row['t2'] = float(line.split(': ')[1])
            elif 'salt_percentage' in line:
                row['salt_percentage'] = float(line.split(': ')[1])
            elif "Epoch" in line:
                accuracy_train.append(float(re.findall(r'Accuracy\:\ \d{2}\.\d+?', line)[0].split()[1]))
                accuracy_test.append(float(re.findall(r'Test Accuracy\:\ \d{2}\.\d+?', line)[0].split()[2]))
            elif "=================" in line:
                row['accuracy_train'] = accuracy_train
                row['accuracy_test'] = accuracy_test
                rows.append(row)
                starts = False

df = pd.DataFrame(rows)
df_temp = df.loc[df['salt_percentage'] == 0.1]
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
for idx, row in df_temp.iterrows():
    if (row['accuracy_test'][-1] > 97) or ((row['t1'] == 1.0) and (row['t2'] == 1.0)):
        ax.plot(row['accuracy_test'], label='t1=%.1f,t2=%.1f'%(row['t1'], row['t2']))
plt.title("Test Accuracy with 10% noise")
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('results', 'TestAccuracePerEpoch.png'))
# plt.show()


df['accuracy_test_last'] = df['accuracy_test'].apply(lambda x:x[-1])
df['accuracy_test_max'] = df['accuracy_test'].apply(max)
for agg, save_dir in zip(['Last', 'Max'], [accuracy_last_dir, accuracy_max_dir]):
    fig = plt.figure(figsize=(7, 7))
    plt.suptitle('Bitempered Loss Evaluation (%s)'%agg)
    plt.tight_layout()
    axes = []
    for ax_idx, salt_percentage in enumerate([0.0, 0.05, 0.1, 0.2]):
        # get salt_percentage data
        df_temp = df.loc[df['salt_percentage'] == salt_percentage]
        t1s, t2s, accuracy_tests = df_temp['t1'].values, df_temp['t2'].values, df_temp['accuracy_test_'+agg.lower()].values / 100

        # plot ax
        ax = fig.add_subplot(2, 2, ax_idx+1, projection='3d')
        ax.scatter3D(t1s, t2s, accuracy_tests, c=accuracy_tests, cmap='viridis')
        ax.set_title('Mnist with ' + str(int(salt_percentage*100)) + '% noise')
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.set_zlabel('Test Accuracy (%s)'%agg)
        axes.append(ax)

    for angle in np.arange(0, 360, 5): # rotate the axes and update
        for ax in axes:
            ax.view_init(elev=30, azim=angle)
        plt.draw()
        # plt.pause(.001)
        plt.savefig(os.path.join(save_dir, str(angle).zfill(3)+".png"))



