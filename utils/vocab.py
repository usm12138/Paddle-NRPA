import collections
import io
import json
import numpy as np
import os
import warnings
from string import punctuation
from collections import Counter
process_dicts={i:'' for i in punctuation}
punc_table = str.maketrans(process_dicts)

def build_vocab(sents, special_tokens=None, descend=False):
    print('==================Building Vocab==================')
    # print(sents)
    # exit()
    text = ' '.join(sent for sent in sents)
    text = text.translate(punc_table)
    words_list= text.lower().split()
    word_dic = dict(Counter(words_list))
    if special_tokens is not None:
        for tok in special_tokens:
            if tok in word_dic:
                del word_dic[tok]
    if descend: # from high to low
        word_dic= {k: v for k, v in sorted(word_dic.items(), key=lambda d:d[1], reverse=True)}
        freqs = list(word_dic.keys())
    else:
        word_dic= {k: v for k, v in sorted(word_dic.items(), key=lambda d:d[1], reverse=False)}
        freqs = list(word_dic.keys())[::-1]
    
    vocab = {k: i+1 for i, k in enumerate(word_dic.keys())}
    vocab['<unk>'] = 0
    print('=======================OVER=====================')
    return vocab, freqs

class Vocab(object):
    def __init__(self,
                 vocab_dict, 
                 counter=None,
                 max_size=None,
                 freq_words=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token=None,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None):
        self._Vocab = vocab_dict
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._freq_words = freq_words
        self._in_ti()
        # Handle special tokens
        combs = (('unk_token', unk_token), ('pad_token', pad_token),
                 ('bos_token', bos_token), ('eos_token', eos_token))
        for name, value in combs:
            self._insert_token(value)
        
    def _in_ti(self):
        self._idx_to_token = {
                idx: token
                for token, idx in self._Vocab.items()
            }
        self._token_to_idx = self._Vocab
    def _insert_token(self, tok):
        if tok not in self._Vocab and tok is not None:
            self._Vocab[tok] = len(self._Vocab)
            self._in_ti()
    @property
    def vocab_size(self):
        return len(self._Vocab)

    def freq_topK(self, topK):
        return self._freq_words[: topK]
    @property
    def unk_token(self):
        return self._unk_token
    @property
    def pad_token(self):
        return self._pad_token
    @property
    def Vocab(self):
        # Returns vocab dict
        return self._Vocab
    @property
    def idx_to_token(self):
        # Returns index-token dict
        return self._idx_to_token

    @property
    def token_to_idx(self):
        # Return token-index dict
        return self._token_to_idx
    
