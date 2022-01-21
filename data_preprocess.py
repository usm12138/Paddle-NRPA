import numpy as np
import pickle
import json
from collections import Counter
from utils.vocab import Vocab, build_vocab
from utils.tokenizer import Tokenizer
from data import *
from utils.utils import *
import nltk
from nltk.corpus import stopwords
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
stopword_list = None
stopword_list= stopwords.words('english')
words = ['no', 'not', 'have', 'has', 'had', 'having', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while']
for w in words:
    stopword_list.remove(w)
# stopword_list.remove('not')
def read_data(fpath):
    users_id, items_id, ratings, reviews, reviews_len = [], [], [], [], []
    print("Reading and processing data..................")
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            info = json.loads(line)
            if len(info['reviewText']) > 600:
                continue
            users_id.append(info['reviewerID'])
            items_id.append(info['asin'])
            ratings.append(info['overall'])
            reviews.append(info['reviewText'])
            reviews_len.append(len(info['reviewText']))
    reviews = [normalize_and_lemmaize(sent) for sent in reviews]
    data = {'users_id':users_id, 'items_id':items_id, 'ratings':ratings, \
            'reviews':reviews, 'len':reviews_len}
    print('Read over!!!!')
    return data

def process(data, tokenizer):
    print('Processing User and Item ID ..................')
    unique_uid, unique_sid = set(data['users_id']), set(data['items_id'])
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    print('The number of user: {}; The number of item: {}'.format(len(user2id), len(item2id)))
    uid = map(lambda x: user2id[x], data['users_id'])
    sid = map(lambda x: item2id[x], data['items_id'])
    data['users_id'] = list(uid)
    data['items_id'] = list(sid)
    print("Tokenizering ..................")
    data['reviews'] = [tokenizer.encode(x, freq_topK=False, stopwords=stopword_list) for x in data['reviews']]
    print("Tokenizer over!!!")
    data_pkl = []
    assert len(data['users_id']) == len(data['items_id']) == len(data['ratings']) == len(data['reviews'])
    for user_id, item_id, rating, review in zip(data['users_id'], data['items_id'], data['ratings'], data['reviews']):
        data_pkl.append({'user_id': user_id, 'item_id': item_id, 'rating': rating, 'review': review})
    return data_pkl


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

def main():
    # fpath = './data/Electronics_5.json'
    # data = read_data(fpath)
    data = pickle.load(open('./data/data_clean.pkl', 'rb'))
    vocab_dict, freqs = build_vocab(data['reviews'], descend=True) # 生成字典
    vocab = Vocab(vocab_dict, freq_words=freqs, unk_token='<unk>', pad_token='<pad>') # 初始化字典
    tokenizer = Tokenizer(vocab) # 分词
    dataset = process(data, tokenizer) 
    u2texts, i2texts = generate_ui2texts(dataset)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
    #保存字典和数据
    pickle.dump(vocab, open('./data/vocab.pkl', 'wb'))
    obj = {'train_dataset': train_dataset, 'valid_dataset': valid_dataset, 'test_dataset': test_dataset, \
    'u2texts': u2texts, 'i2texts': i2texts}
    pickle.dump(obj, open('./data/data.pkl', 'wb'))
if __name__ == '__main__':
    main()


