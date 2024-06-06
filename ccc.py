
1
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


































2
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

















3
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













4
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



5
class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        # self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens, src_mask=None, sent_positions=None, sent_nums=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """

        src_embed = self.embedding(src)  # [batch, src_len, embed_size]

        # sort src_embed according to its length
        batch_size = src.size(0)
        assert len(src_lens) == batch_size
        sort_ind = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens_sorted = [src_lens[i] for i in sort_ind]
        src_embed = reorder_sequence(src_embed, sort_ind, batch_first=True)
        # src_embed_sorted_np = src_embed.detach().cpu().numpy()[:, 0, :]

        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens_sorted, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # restore the order of memory_bank
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(batch_size)]
        memory_bank = reorder_sequence(memory_bank, reorder_ind, batch_first=True)
        encoder_final_state = reorder_gru_states(encoder_final_state, reorder_ind)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)  # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return (memory_bank.contiguous(), None), (encoder_last_layer_final_state, None)


def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first, [B, T, D] if batch first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    device = sequence_emb.device
    order = torch.LongTensor(order)
    order = order.to(device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_


def reorder_gru_states(gru_state, order):
    """
    gru_state: [num_layer * num_directions, batch, hidden_size]
    order: list of sequence length
    """
    assert len(order) == gru_state.size(1)
    order = torch.LongTensor(order).to(gru_state.device)
    sorted_state = gru_state.index_select(index=order, dim=1)
    return sorted_state


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn,
                 review_attn, pad_idx, attn_mode, dropout=0.0, rating_memory_pred=False):
        super(RNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.review_attn = review_attn
        self.pad_token = pad_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.input_size = embed_size
        self.hr_enc = False

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=False, dropout=dropout)
        self.attention_layer = Attention(
            decoder_size=hidden_size,
            memory_bank_size=memory_bank_size,
            coverage_attn=coverage_attn,
            attn_mode=attn_mode
        )

        merged_memory_bank_size = memory_bank_size

        self.rating_memory_pred = rating_memory_pred
        if self.rating_memory_pred:
            self.rating_attention_layer = Attention(
                decoder_size=hidden_size,
                memory_bank_size=embed_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode)
            merged_memory_bank_size += embed_size

        if copy_attn:
            p_gen_input_size = embed_size + hidden_size + merged_memory_bank_size
            """
            if goal_vector_mode == 2:
                p_gen_input_size += goal_vector_size
            """
            self.p_gen_linear = nn.Linear(p_gen_input_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.vocab_dist_linear_1 = nn.Linear(hidden_size + merged_memory_bank_size, hidden_size)
        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, h, memory_banks, src_masks, max_num_oovs, src_oov, coverage, decoder_memory_bank=None,
                rating_memory_bank=None, goal_vector=None):
        """
        :param y: [batch_size]
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_banks: ([batch_size, max_src_seq_len, memory_bank_size], None)
        :param src_masks: ([batch_size, max_src_seq_len], None)
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :param decoder_memory_bank: [batch_size, t-1, decoder_size]
        :param rating_memory_bank: a FloatTensor, [batch, rating_v_size, emb_size]
        :param goal_vector: [1, batch_size, goal_vector_size]
        :return:
        """
        # use a consistent interface with HirEncRNNDecoder
        memory_bank = memory_banks[0]
        src_mask = src_masks[0]

        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]

        rnn_input = y_emb

        _, h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        word_context, word_attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask,
                                                                      coverage)
        # context: [batch_size, memory_bank_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        assert word_context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert word_attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

        context = word_context
        vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)  # [B, memory_bank_size + decoder_size]
        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))

        p_gen = None
        if self.copy_attn:
            """
            if self.goal_vector_mode == 2:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), goal_vector.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size + goal_vector]
            else:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
            """
            p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)), dim=1)
            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * word_attn_dist

            if max_num_oovs > 0:
                # extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return final_dist, h_next, word_context, word_attn_dist, p_gen, coverage











6
class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = len(opt.word2idx)
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size

        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout
        self.model_type = opt.model_type

        self.bridge = opt.bridge

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = io.PAD
        self.pad_idx_trg = io.PAD
        self.bos_idx = io.BOS
        self.eos_idx = io.EOS
        self.unk_idx = io.UNK
        self.sep_idx = None

        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn

        self.attn_mode = opt.attn_mode

        self.rating_memory_pred = opt.rating_memory_pred

        self.encoder = RNNEncoderBasic(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )

        self.decoder = RNNDecoder(
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            review_attn=self.review_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = nn.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, 'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        if self.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        self.init_embedding_weights()

    def init_embedding_weights(self):
        """Initialize weights."""
        init_range = 0.1
        self.encoder.embedding.weight.data.uniform_(-init_range, init_range)
        if not self.share_embeddings:
            self.decoder.embedding.weight.data.uniform_(-init_range, init_range)

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self.share_embeddings

        assert self.encoder.embedding.weight.size() == embedding.size()
        self.encoder.embedding.weight.data.copy_(embedding)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums,
                src_sent_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param source_representation_target_list: a list that store the index of ground truth source representation for each batch, dim=[batch_size]
        :param src_sent_positions: a LongTensor containing the forward and backward ending positions of src sentences, [batch, max_sent_num, 2]
        :param src_sent_nums: a list containing the sentence number of each src, [batch]
        :param src_sent_mask: a FloatTensor, [batch, max_sent_num]
        :return:
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        memory_banks, encoder_final_states = self.encoder(src, src_lens, src_mask, src_sent_positions, src_sent_nums)
        word_memory_bank, sent_memory_bank = memory_banks
        word_encoder_final_state, sent_encoder_final_state = encoder_final_states
        src_masks = (src_mask, src_sent_mask)

        assert word_memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert word_encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # Decoding
        h_t_init = self.init_decoder_state(word_encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)

        decoder_dist_all = []
        attention_dist_all = []

        coverage = None
        coverage_all = None
        decoder_memory_bank = None

        # init y_t to be BOS token
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]

        for t in range(max_target_length):
            # determine the hidden state that will be feed into the next step
            # according to the time step or the target input
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, h_t, memory_banks, src_masks, max_num_oov, src_oov, coverage, decoder_memory_bank)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]

            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        return decoder_dist_all, h_t_next, attention_dist_all, word_encoder_final_state, coverage_all

    def tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
        """
        :param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
        :param batch_size:
        :param hidden_size:
        :param seq_lens: a list that store the seq len of each batch, with len=batch_size
        :return: [batch_size, hidden_size, max_seq_len]
        """
        # assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
        max_seq_len = max(seq_lens)
        for i in range(batch_size):
            for j in range(max_seq_len - seq_lens[i]):
                tensor_2d_list[i].append(torch.ones(hidden_size).to(self.device) * self.pad_idx_trg)  # [hidden_size]
            tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=1)  # [hidden_size, max_seq_len]
        tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, hidden_size, max_seq_len]
        return tensor_3d

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state



















7
    parser = argparse.ArgumentParser(description='train_ml.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.train_ml_opts(parser)
    opt = parser.parse_args(args=[])
    opt = train_process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")










8
def build_loader(data_path, batch_size, word2idx, src_max_len, trg_max_len, num_workers):
    coll_fn_customized = io.summ_rating_flatten_coll_fn(word2idx=word2idx, src_max_len=src_max_len,
                                                        trg_max_len=trg_max_len)

    train_loader = DataLoader(SummRating('train', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=True)

    valid_loader = DataLoader(SummRating('val', data_path), collate_fn=coll_fn_customized, num_workers=num_workers,
                              batch_size=batch_size, pin_memory=True, shuffle=False)
    return train_loader, valid_loader





9
def evaluate_loss(data_loader, overall_model, opt):
    overall_model.eval()
    generation_loss_sum = 0.0
    total_trg_tokens = 0
    total_num_iterations = 0

    batch_number = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            batch_number += 1
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src_mask = batch['src_mask']
            src_sent_positions = batch['src_sent_positions']
            src_sent_nums = batch['src_sent_nums']
            src_sent_mask = batch['src_sent_mask']
            src_oov = batch['src_oov_tensor']
            oov_lists = batch['oov_lists']
            src_str_list = batch['src_list_tokenized']
            trg_sent_2d_list = batch['tgt_sent_2d_list']
            trg = batch['tgt_tensor']
            trg_oov = batch['tgt_oov_tensor']
            trg_lens = batch['tgt_lens']
            trg_mask = batch['tgt_mask']

            indices = batch['original_indices']

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            total_num_iterations += 1
            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_sent_positions = src_sent_positions.to(opt.device)
            src_sent_mask = src_sent_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)
            decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage = overall_model(
                src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums,
                src_sent_mask)

            if decoder_dist is not None:
                if opt.copy_attention:  # Compute the loss using target with oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                                           opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                           opt.lambda_coverage, coverage_loss=False)
                else:  # Compute the loss using target without oov words
                    generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                                           opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                           opt.lambda_coverage, coverage_loss=False)
            else:
                generation_loss = torch.Tensor([0.0]).to(opt.device)

            # normalize generation loss
            num_trg_tokens = sum(trg_lens)
            normalized_generation_loss = generation_loss.div(num_trg_tokens)

            generation_loss_sum += normalized_generation_loss.item()

    return normalized_generation_loss.item() / batch_number


import math
import torch
import os

EPS = 1e-8


10定义训练函数
def train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt):
    print('======================  Start Training  =========================')

    total_batch = 0
    early_stop_flag = False

    best_valid_joint_loss = float('inf')
    num_stop_dropping = 0

    previous_valid_loss = float('inf')
    overall_model.train()

    for epoch in range(0, opt.epochs):
        if early_stop_flag:
            break

        for batch_i, batch in enumerate(train_data_loader):
            if early_stop_flag:
                break
            total_batch += 1

            # Training
            train_loss = train_one_batch(batch, overall_model, optimizer_ml, opt)
            if total_batch % opt.checkpoint_interval == 0:
                print('epoch:', epoch, '  ', 'total_batch', total_batch)
            if epoch >= 1:
                if total_batch % opt.checkpoint_interval == 0:
                    print('start valid')
                    valid_loss = evaluate_loss(valid_data_loader, overall_model, opt)
                    print('valid_loss', valid_loss)
                    overall_model.train()

                    if epoch >= opt.start_decay_and_early_stop_at:
                        if valid_loss < previous_valid_loss:  # update the best valid loss and save the model parameters
                            previous_valid_loss = valid_loss
                            print("Valid loss drops")

                            num_stop_dropping = 0
                            check_pt_model_path = os.path.join(opt.model_path, 'train_model')
                            torch.save(  # save model parameters
                                overall_model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            print('Saving checkpoint to %s' % check_pt_model_path)
                        else:
                            num_stop_dropping += 1
                            # decay the learning rate by a factor
                            if opt.learning_rate_decay < 1:
                                for i, param_group in enumerate(optimizer_ml.param_groups):
                                    old_lr = float(param_group['lr'])
                                    new_lr = old_lr * opt.learning_rate_decay
                                    if new_lr < opt.min_lr:
                                        new_lr = opt.min_lr
                                    if old_lr - new_lr > EPS:
                                        param_group['lr'] = new_lr

                        if not opt.disable_early_stop:
                            print("num_stop_dropping", num_stop_dropping)
                            if num_stop_dropping >= opt.early_stop_tolerance:
                                print('Have increased for %d check points, early stop training' % num_stop_dropping)
                                early_stop_flag = True
                                break


def train_one_batch(batch, overall_model, optimizer, opt):
    src = batch['src_tensor']
    src_lens = batch['src_lens']
    src_mask = batch['src_mask']
    src_sent_positions = batch['src_sent_positions']
    src_sent_nums = batch['src_sent_nums']
    src_sent_mask = batch['src_sent_mask']
    src_oov = batch['src_oov_tensor']
    oov_lists = batch['oov_lists']
    src_str_list = batch['src_list_tokenized']
    trg_sent_2d_list = batch['tgt_sent_2d_list']
    trg = batch['tgt_tensor']
    trg_oov = batch['tgt_oov_tensor']
    trg_lens = batch['tgt_lens']
    trg_mask = batch['tgt_mask']

    indices = batch['original_indices']
    """
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[io.SEP_WORD]
                 if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)

    src_sent_positions = src_sent_positions.to(opt.device)
    src_sent_mask = src_sent_mask.to(opt.device)

    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    optimizer.zero_grad()

    decoder_dist, h_t, seq2seq_attention_dist, encoder_final_state, coverage = \
        overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums,
                      src_sent_mask)

    if decoder_dist is not None:
        if opt.copy_attention:  # Compute the loss using target with oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                                   opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                   opt.lambda_coverage, opt.coverage_loss)
        else:  # Compute the loss using target without oov words
            generation_loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                                   opt.coverage_attn, coverage, seq2seq_attention_dist,
                                                   opt.lambda_coverage, opt.coverage_loss)
    else:

        generation_loss = torch.Tensor([0.0]).to(opt.device)

    if math.isnan(generation_loss.item()):
        raise ValueError("Generation loss is NaN")

    # normalize generation loss
    total_trg_tokens = sum(trg_lens)
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        generation_loss_normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        generation_loss_normalization = batch_size
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert generation_loss_normalization > 0, 'normalization should be a positive number'
    normalized_generation_loss = generation_loss.div(generation_loss_normalization)
    # back propagation on the joint loss
    normalized_generation_loss.backward()
    optimizer.step()
    return normalized_generation_loss.item()















建立词典，读取数据，初始化Seq2SeqModel。将模型放到GPU上去，打印模型，训练
# construct vocab
with open(join(opt.data, 'vocab_cnt.pkl'), 'rb') as f:
    wc = pkl.load(f)
word2idx, idx2word = io.make_vocab(wc, opt.v_size)
opt.word2idx = word2idx
opt.idx2word = idx2word
opt.vocab_size = len(word2idx)
# construct train_data_loader, valid_data_loader

train_data_loader, valid_data_loader = build_loader(opt.data, opt.batch_size, word2idx, opt.src_max_len, opt.trg_max_len, opt.batch_workers)

# construct model

overall_model = Seq2SeqModel(opt)
overall_model.to(opt.device)

# construct optimizer
optimizer_ml = torch.optim.Adam(params=filter(lambda p: p.requires_grad, overall_model.parameters()), lr=opt.learning_rate)

print(overall_model)
train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt)#如果只是想预测，可加载训练好的模型（只保存了模型参数），注释该行。





















11建立词典，读取数据，初始化Seq2SeqModel。将模型放到GPU上去，打印模型，训练
from src.io import DecodeDataset, eval_coll_fn, SummRating
from src.predict import predict
test_loader = DataLoader(DecodeDataset("test", opt.data), collate_fn=eval_coll_fn(word2idx=word2idx, src_max_len=800),
                                  num_workers=opt.batch_workers,
                                  batch_size=16, pin_memory=True, shuffle=False)

overall_model.load_state_dict(torch.load('./models/train_model'))
overall_model.eval()
all_sample=predict(test_loader,overall_model,opt)
all_sample_test=[]
for i in range(len(all_sample)):
    all_sample_test.append([str(i),all_sample[i]])
with open(join("./result", 'submission.csv'),'w+') as csvfile:
    writer=csv.writer(csvfile,delimiter="\t")
    writer.writerows(all_sample_test)

    import pandas as pd
    from rouge import Rouge

    pd_output = pd.read_csv("./result/submission.csv", sep="\t", names=["index", "output"])
    pd_label = pd.read_csv("./datasets/test_label.csv", sep="\t", names=["index", "label"])

    output = pd_output.output
    label = pd_label.label

    rouge = Rouge()
    rouge_score = rouge.get_scores(output, label)

    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))