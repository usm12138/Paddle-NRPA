from collections import Counter

def get_idx_from_word(word, word_to_idx, unk_word):
    if word in word_to_idx:
        return word_to_idx[word]
    return word_to_idx[unk_word]
    
class BaseTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_tokenizer(self):
        return self.tokenizer

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass

from string import punctuation
process_dicts={i:'' for i in punctuation}
punc_table = str.maketrans(process_dicts)
def transformations(text, freq_topK=None, stopwords=None):
    import re
    text = re.sub('<.*?>', '', text) #HTML 标签移除
    text = text.translate(punc_table) #去除标点
    words = text.lower().split() # 小写转换
    return [x for x in words if (not x.isdigit() and (freq_topK is None or x not in freq_topK)\
             and (stopwords is None or x not in stopwords))] #数字,高频词和停用词去除

class Tokenizer(BaseTokenizer):
    def __init__(self, vocab):
        self.vocab = vocab
    def encode(self, sentence, freq_topK=False, stopwords=None):
        if freq_topK:
            words = transformations(sentence, freq_topK=self.vocab.freq_topK(15), stopwords=stopwords)
        else:
            words = transformations(sentence, stopwords=stopwords)
        return [get_idx_from_word(word, self.vocab.token_to_idx, \
                    self.vocab.unk_token) for word in words]
