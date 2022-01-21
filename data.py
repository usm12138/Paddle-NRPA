
import numpy as np
import json
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
import random
from utils.tokenizer import Tokenizer

def process(data):
    user_ids, item_ids, ratings = [], [], []
    for x in data:
        user_ids.append(x['user_id'])
        item_ids.append(x['item_id'])
        ratings.append(x['rating'])
    return user_ids, item_ids, ratings
def load_data(fpath):
    obj = pickle.load(open(fpath, 'rb'))
    return obj['train_dataset'], obj['valid_dataset'], obj['test_dataset'], obj['u2texts'], obj['i2texts']
# def load_data(fpath):
#     data = pickle.load(open(fpath, 'rb'))
#     train_data, valid_data, test_data = split_dataset(data)
#     # u_train, i_train, r_train = process(train_data)
#     # u_valid, i_valid, r_valid = process(valid_data)
#     # u_test, i_test, r_test = process(test_data)
#     # return process(train_data), process(valid_data), process(test_data)
#     return train_data, valid_data, test_data

def split_dataset(data):
    s1 = int(len(data) * 0.8)
    s2 = int(len(data) * 0.9)
    random.shuffle(data)
    random.shuffle(data)
    # random.shuffle(data)
    # random.shuffle(data)
    return data[: s1], data[s1: s2], data[s2: ]

def load_vocab(fpath):
    return pickle.load(open(fpath, 'rb'))

def generate_ui2textIDs(dataset):
    '''生成用户和物品所有对应的评论, 字典保存'''
    user_reviews, item_reviews = dict(), dict()
    textIDs = []
    for i, x in enumerate(dataset):
        textIDs.append(x['review'])
        if x['user_id'] in user_reviews:
            user_reviews[x['user_id']].append(i)
        else:
            user_reviews[x['user_id']] = [i]
        if x['item_id'] in item_reviews:
            item_reviews[x['item_id']].append(i)
        else:
            item_reviews[x['item_id']] = [i]
    return user_reviews, item_reviews, textIDs

def generate_ui2texts(dataset):
    '''生成用户和物品所有对应的评论, 字典保存'''
    user_reviews, item_reviews = dict(), dict()
    for x in dataset:
        if x['user_id'] in user_reviews:
            user_reviews[x['user_id']].append(x['review'])
        else:
            user_reviews[x['user_id']] = [x['review']]
        if x['item_id'] in item_reviews:
            item_reviews[x['item_id']].append(x['review'])
        else:
            item_reviews[x['item_id']] = [x['review']]
    return user_reviews, item_reviews

def load_ui2texts(fpath):
    ui2texts = pickle.load(open(fpath, 'rb'))
    return ui2texts['user_reviews'], ui2texts['item_reviews']

# def statistic(ui2textIDs, reviews):
#     seq_len, seq_size = [], []
#     for k, v in ui2texts.items():
#         for x in v:
#             seq_len.append(len(x))
#         seq_size.append(len(v))
#     seq_len.sort(), seq_size.sort()
#     print('最大序列长度: ', max(seq_len), '最小序列长度: ', min(seq_len), \
#     '序列长度的平均: ', np.mean(np.array(seq_len)), '序列长度的中位数: ', seq_len[int(len(seq_len)*0.9)])
#     print('最大序列数量: ', max(seq_size), '最小序列数量: ', min(seq_size), \
#     '序列长度的平均: ', np.mean(np.array(seq_size)), '序列数量的中位数: ', seq_size[int(len(seq_size)*0.9)])
def statistic(ui2texts):
    seq_len, seq_size = [], []
    for k, v in ui2texts.items():
        for x in v:
            seq_len.append(len(x))
        seq_size.append(len(v))
    seq_len.sort(), seq_size.sort()
    print('最大序列长度: ', max(seq_len), '最小序列长度: ', min(seq_len), \
    '序列长度的平均: ', np.mean(np.array(seq_len)), '前90%的序列长度: ', seq_len[int(len(seq_len)*0.9)])
    print('最大序列数量: ', max(seq_size), '最小序列数量: ', min(seq_size), \
    '序列长度的平均: ', np.mean(np.array(seq_size)), '前90%的序列数量: ', seq_size[int(len(seq_size)*0.9)])

def padding(texts, vocab, max_len, max_size):
    sen_indices = []
    for sen in texts:
        if len(sen) < max_len:
            num_padding = max_len - len(sen)
            sen += [vocab.token_to_idx['<pad>']] * num_padding
        sen_indices.append(sen[: max_len])
    if max_size > len(sen_indices):
        num_padding = max_size - len(sen_indices)
        sen_indices += [[vocab.token_to_idx['<pad>']] * max_len] * num_padding  
    import random
    random.shuffle(sen_indices)
    return sen_indices[: max_size]


class Dataset(paddle.io.Dataset):
    def __init__(self, uids, iids, ratings, u2texts, i2texts, vocab, args):
        self.uids = uids
        self.iids = iids
        self.ratings =ratings
        self.u2texts = u2texts
        self.i2texts = i2texts
        self.vocab = vocab
        self.args = args

    def __getitem__(self, index):       
        utexts = self.u2texts[self.uids[index]]
        itexts = self.i2texts[self.iids[index]]
        return self.uids[index], self.iids[index], self.ratings[index],\
         padding(utexts, self.vocab, self.args.u2r_max_seq_len, self.args.u2r_max_seq_size),\
         padding(itexts, self.vocab, self.args.i2r_max_seq_len, self.args.i2r_max_seq_size)
        
    def __len__(self):
        return len(self.ratings)
