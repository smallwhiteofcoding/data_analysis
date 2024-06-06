import json
import random

train_path = './datasets/train.csv'  # 自定义训练集路径
with open(train_path, 'r', encoding='utf-8') as f:
    train_data_all = f.readlines()

test_path = './datasets/test.csv'  # 自定义测试集路径
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = f.readlines()
random.shuffle(train_data_all)

new_data_train_path = './datasets/new_data/train/'
new_data_val_path = './datasets/new_data/val/'
new_data_test_path = './datasets/new_data/test/'

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



































from os.path import join
import json
import os
import random
from collections import Counter
import pickle as pkl
import re
def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def bulid_vocab_counter(data_dir):
    split_dir = join(data_dir, "train")
    n_data = _count_data(split_dir)
    vocab_counter = Counter()
    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        summary = js['summary']
        summary_text = ' '.join(summary).strip()
        summary_word_list = summary_text.strip().split(' ')

        review = js['source']
        review_text = ' '.join(review).strip()
        review_word_list = review_text.strip().split(' ')

        all_tokens = summary_word_list + review_word_list
        vocab_counter.update([t for t in all_tokens if t != ""])

    with open(os.path.join(data_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
bulid_vocab_counter('./datasets/new_data')


















import torch
import argparse
from src import config
import os
from os.path import join
import json
from src import io
from src.io import SummRating
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle as pkl
import numpy as np
from collections import Counter
import re
import torch.nn as nn
from src.attention import Attention
from src.masked_softmax import MaskedSoftmax
import random
import datetime
import time
import math
import csv
from src.masked_loss import masked_cross_entropy














def train_process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    opt.exp += '.ml'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if "ordinal" in opt.classifier_loss_type:
        opt.ordinal = True
    else:
        opt.ordinal = False

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    return opt