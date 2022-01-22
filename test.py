import numpy as np
import argparse
import pickle
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.nn.functional as F
import random
import time
import os

from models.NRPA import NRPA

from data import *
paddle.device.set_device('gpu:1')
device = paddle.CUDAPlace(1)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data', type=str)
    parser.add_argument("--model_dir", default='./model', type=str)
    parser.add_argument("--seed", default=6666, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--conv_kernel_num", default=80, type=int)
    parser.add_argument("--atten_vec_dim", default=80, type=int)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--id_embedding_dim", default=32, type=int)
    parser.add_argument("--fm_k", default=10, type=int)
    parser.add_argument("--latent_factor_num", default=32, type=int)
    parser.add_argument("--word_vec_dim", default=300, type=int)
    parser.add_argument("--u2r_max_seq_size", default=11, type=int)
    parser.add_argument("--u2r_max_seq_len", default=52, type=int)
    parser.add_argument("--i2r_max_seq_size", default=35, type=int)
    parser.add_argument("--i2r_max_seq_len", default=52, type=int)
    args = parser.parse_args()
    return args
def eval(model, data_loader):
    error = []
    model.eval()
    for u_ids, i_ids, ratings, u_texts, i_texts in data_loader:
        with paddle.no_grad():
            u_ids = paddle.to_tensor(u_ids, place=device)
            i_ids = paddle.to_tensor(i_ids, place=device)
            ratings = paddle.to_tensor(ratings, place=device, dtype='float32')
            u_texts = paddle.to_tensor(u_texts, place=device).transpose([2, 0, 1])
            i_texts = paddle.to_tensor(i_texts, place=device).transpose([2, 0, 1])
            batch_pred = model(u_texts, i_texts, u_ids, i_ids)
            batch_error = batch_pred - ratings
            error.append(batch_error.cpu().numpy())
    error = np.concatenate(error, axis=None)**2
    error = error.mean().item()
    return error

def main(args):
    print('Loading and processing data..............')
    #加载已预处理的数据
    train_dataset, valid_dataset, test_dataset, u2texts, i2texts = load_data(os.path.join(args.data_dir, 'data.pkl')) 
    vocab = load_vocab(os.path.join(args.data_dir, 'vocab.pkl'))#加载词典
    u_train, i_train, r_train = process(train_dataset)
    u_valid, i_valid, r_valid = process(valid_dataset)
    u_test, i_test, r_test =  process(test_dataset)
    print('Loading and processing complete!!!!!!!!!!!')
    user_num, item_num = max(u_train+u_valid+u_test) + 1, max(i_train+i_valid+i_test) + 1
    print('Loading model..............')
    #加载已训练好的模型
    model = NRPA(user_num, item_num, vocab, args)
    model_state = paddle.load(os.path.join(args.model_dir, 'model.pdparams'))
    model.load_dict(model_state)
    
    test_data_loader = DataLoader(
           Dataset(u_test, i_test, r_test, u2texts, i2texts, vocab, args),
           batch_size=args.batch_size,
           num_workers=4
        )
    t0 = time.time()
    print('Testing..............')
    #测试模型
    loss_test = eval(model, test_data_loader)
    print('test mse_loss: {:.5f}, time: {:.4f} s'.format(loss_test, time.time() - t0))
    print()

if __name__ == '__main__':
    args = parse_args()
    main(args)