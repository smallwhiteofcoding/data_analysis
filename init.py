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


#设置相关参数
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