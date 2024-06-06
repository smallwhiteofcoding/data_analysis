#先将下载的数据进行处理，把训练集划分为训练子集与验证集（没有固定随机种子，每次划分不一样），另行存储，方便后续处理。
import json
import os
import random
train_path = 'datasets/train_dataset.csv'  # 自定义训练集路径
with open(train_path, 'r', encoding='utf-8') as f:
    train_data_all = f.readlines()

test_path = 'datasets/test_dataset.csv'  # 自定义测试集路径
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = f.readlines()
random.shuffle(train_data_all)

new_data_train_path = 'datasets/new_data/train/'
new_data_val_path = 'datasets/new_data/val/'
new_data_test_path = 'datasets/new_data/test/'

if not os.path.exists(new_data_train_path):
    os.makedirs(new_data_train_path)

if not os.path.exists(new_data_val_path):
    os.makedirs(new_data_val_path)

if not os.path.exists(new_data_test_path):
    os.makedirs(new_data_test_path)

train_data = train_data_all[:8000]  # 把训练集重新划分为训练子集和验证子集，保证验证集上loss最小的模型，预测测试集
val_data = train_data_all[8000:]

j = 0
for i in range(len(train_data)):

    if len(train_data[i].split('\t')) == 3:

        source_seg = train_data[i].split("\t")[1]
        traget_seg = train_data[i].split("\t")[2].strip('\n')
        if source_seg and traget_seg != '':
            dict_data = {"source": [source_seg], "summary": [traget_seg]}
            with open(new_data_train_path + str(j) + '.json', 'w+', encoding='utf-8') as f:  # 自定义保存训练集文件路径
                f.write(json.dumps(dict_data, ensure_ascii=False))
            j += 1
j = 0
for i in range(len(val_data)):

    if len(val_data[i].split('\t')) == 3:

        source_seg = val_data[i].split("\t")[1]
        traget_seg = val_data[i].split("\t")[2].strip('\n')
        if source_seg and traget_seg != '':
            dict_data = {"source": [source_seg], "summary": [traget_seg]}

            with open(new_data_val_path + str(j) + '.json', 'w+', encoding='utf-8') as f:  # 自定义保存训练集文件路径
                f.write(json.dumps(dict_data, ensure_ascii=False))
            j += 1

j = 0
for i in range(len(test_data)):

    source = test_data[i].split('\t')[1].strip('\n')

    if source != '':
        dict_data = {"source": [source], "summary": ['no summary']}  # 测试集没有参考摘要，提交预测结果文件后，计算分数
        with open(new_data_test_path + str(j) + '.json', 'w+', encoding='utf-8') as f:  # 自定义保存验证集文件路径
            f.write(json.dumps(dict_data, ensure_ascii=False))
        j += 1