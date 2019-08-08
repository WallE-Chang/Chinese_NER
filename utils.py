# -*- coding: utf-8 -*-
# @Author: wanli
# @Date:   2019-07-24 18:02:50
# @Last Modified by:   wanli
# @Last Modified time: 2019-08-08 16:36:37

import math
import numpy as np

from typing import List
from typing import Tuple



def read_corpus(corpus_path: str)->List[List[str]]:
    '''read corpus and return the list of sentence and targets

    Arguments:
        corpus_path {str} -- corpus path

    Returns:
        List[List[str],List[str]] -- list of sentence and targets
    '''
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    
    return data


def batch_iter(data: List[List], batch_size: int, shuffle=False) -> Tuple[List]:
    '''Yield batches of source and target sentences reverse sorted by length (largest to smallest).

    Arguments:
        data {List[List[str],List[str]]} -- list of tuples containing sentences and labels
        batch_size {int} -- batch size

    Keyword Arguments:
        shuffle {bool} -- whether to randomly shuffle the dataset (default: {False})

    Yields:
        Tuple[List,List] -- batches of source and target sentences
    '''
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        x = [e[0] for e in examples]
        y = [e[1] for e in examples]

        yield x, y


def load_embedding(path: str)->Tuple[dict, np.array]:
    '''load pretrain word embeeding  and return (word_to_index,index_to_vector)
    Note: pad token was added


    Arguments:
        path {str} -- pretrain word embeeding

    Returns:
        Tuple[dict,np.numpy] -- (word_to_index,index_to_vector)
    '''
    word_to_index = {}
    index_to_vector = []
    f = open(path, encoding='utf8')
    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_to_index[word] = index
        index_to_vector.append(vector)
        # if index >=3000000:
        # # if index >=3000:
        #     break
    f.close()
    word_to_index['<pad>'] = 0
    index_to_vector.insert(word_to_index['<pad>'], np.zeros(vector.shape))
    index_to_vector = np.stack(index_to_vector)
    return word_to_index, index_to_vector
