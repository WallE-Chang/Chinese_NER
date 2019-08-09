# -*- coding: utf-8 -*-
# @Author: wanli
# @Date:   2019-07-25 09:21:59
# @Last Modified by:   changwanli
# @Last Modified time: 2019-08-09 14:14:58

import torch

from config import Const
from typing import Dict
from typing import List
from typing import Tuple


class data_preprocess(object):
    """For X (input sentences) : padding + word2index + index2vector
       For Y (input tages): padding + tag2label
    """

    def __init__(self, training_data: List[List[str]], pretrained_word_to_index: Dict[str, int]=None):
        '''initialize data_preprocess

        Arguments:
            training_data {List[List[str]]} -- List[sentence of words,sentence of tags]

        Keyword Arguments:
            pretrained_word_to_index {Dict[str,int]} -- if model use pretrained word embedding, pretrained_word_to_index == the dict of word2index
                                                        else pretrained_word_to_index = None (default: {None})
        '''

        self.use_pretrained_word_to_index = False
        if pretrained_word_to_index is not None:
            self.use_pretrained_word_to_index = True
        self.word_to_index, self.tag_to_index = self.__vocab_build(
            training_data, pretrained_word_to_index=pretrained_word_to_index)
        self.idx_to_tag = {ix: tag for tag, ix in self.tag_to_index.items()}
        print("Tag vocab:", self.tag_to_index)
        print(f'The number of vacob: {len(self.word_to_index)}')

    def __vocab_build(self, training_data: List[List[str]], min_count=1, pretrained_word_to_index: Dict[str, int]=None) -> Tuple[Dict[str, int]]:
        '''generate word_to_index and tag_to_index according to training_data

        Arguments:
            training_data {List[List[str]]} -- List[sentence of words,sentence of tags]

        Keyword Arguments:
            min_count {number} -- if word occurs n < freq_cutoff times, drop the word (default: {2})
            pretrained_word_to_index {Dict[str, int]} -- if model use pretrained word embedding, pretrained_word_to_index == the dict of word2index
                                                         else pretrained_word_to_index = None (default: {None})
        Returns:
            Tuple[Dict[str,int]] -- (word_to_index, tag_to_index)
        '''

        if not self.use_pretrained_word_to_index:
            word_to_index = {}
        tag_to_index = {
            Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID,
            Const.BOS_TAG_TOKEN: Const.BOS_TAG_ID,
            Const.EOS_TAG_TOKEN: Const.EOS_TAG_ID,
        }

        for sentence, tags in training_data:
            for word, tag in zip(sentence, tags):
                if not self.use_pretrained_word_to_index:
                    if word.isdigit():
                        word = '<num>'
                    elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                        word = '<eng>'
                    if word not in word_to_index:
                        word_to_index[word] = [len(word_to_index) + 1, 1]
                    else:
                        word_to_index[word][1] += 1

                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)

        if not self.use_pretrained_word_to_index:
            low_freq_words = []
            for word, [word_id, word_freq] in word_to_index.items():
                if word_freq < min_count and word != '<num>' and word != '<eng>':
                    low_freq_words.append(word)
            for word in low_freq_words:
                del word_to_index[word]

            new_id = 2
            for word in word_to_index.keys():
                word_to_index[word] = new_id
                new_id += 1
            word_to_index[Const.UNK_TOKEN] = Const.UNK_ID
            word_to_index[Const.PAD_TOKEN] = Const.PAD_ID
        else:
            word_to_index = pretrained_word_to_index

        return word_to_index, tag_to_index

    @staticmethod
    def pad_sents(sents_idx: List[List[int]], pad_id: int) ->Tuple[torch.Tensor, torch.FloatTensor]:
        '''Pad list of sentences according to the longest sentence in the batch.

        Arguments:
            sents_idx {List[List[int]]} -- list of sentences, where each sentence
                              is represented as a list of word index
            pad_id {int} -- padding index

        Returns:
             Tuple[torch.Tensor,torch.FloatTensor] -- (sents_idx_padded,mask)

             sents_idx_padded is a torch.Tensor with shape of (batch_size,max_len) where sentences shorter than the
             max length sentence are padded out with the pad_id, such that each sentences in the batch
             now has equal length

             mask is a tensor representing valid positions.  Shape:(batch_size, seq_len).
             pad position = 0, valid position = 1
        '''
        batch_size = len(sents_idx)

        max_len = max(list(map(len, sents_idx)))

        sents_idx_padded = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long)
        for i in range(len(sents_idx_padded)):
            sents_idx_padded[i, :len(sents_idx[i])] = torch.tensor(
                sents_idx[i], dtype=torch.float)

        mask = (sents_idx_padded != pad_id).float()
        return sents_idx_padded, mask

    def sents2indices(self, sents: List[List[str]]) ->List[List[int]]:
        '''Convert list of sentences of words into list of list of indices.


        Arguments:
            sents {List[List[str]]} -- sentences in words

        Returns:
            List[List[int]] -- sentences in indices
        '''

        sents_idx = []
        for sent in sents:
            sent_idx = []
            for word in sent:
                if self.use_pretrained_word_to_index:
                    if word not in self.word_to_index:
                        word = 'unk'
                else:
                    if word.isdigit():
                        word = '<num>'
                    elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                        word = '<eng>'
                    if word not in self.word_to_index:
                        word = '<unk>'
                sent_idx.append(self.word_to_index[word])
            sents_idx.append(sent_idx)
        return sents_idx

    def to_input_idx(self, sents: List[List[str]], device: torch.device) -> Tuple[torch.Tensor, torch.FloatTensor]:
        '''Convert list of sentences  into embedding tensor with necessary padding for 
        shorter sentences.


        Arguments:
            sents {List[List[str]]} -- list of sentences
            device {torch.device} -- device on which to load the tesnor, i.e. CPU or GPU

        Returns:
            Tuple[torch.Tensor,torch.FloatTensor] -- (sents_padded,mask)
            sents_padded is tensor of (batch_size,max_sentence_length,d_embedding)
            mask is tensor of (batch_size,max_sentence_length)
        '''
        sents_idx = self.sents2indices(sents)
        sents_idx_padded, mask = self.pad_sents(sents_idx, Const.PAD_ID)
        sents_idx_padded = sents_idx_padded.to(device)
        mask = mask.to(device)
        return sents_idx_padded, mask

    def to_input_label(self, tags: List[List[str]], device: torch.device) -> torch.Tensor:
        '''Convert list of sentences of tags into labels with necessary padding for 
        shorter sentences.


        Arguments:
            tags {List[List[str]]} -- [list of sentences of tags]
            device {torch.device} --  device on which to load the tesnor, i.e. CPU or GPU

        Returns:
            torch.Tensor --  [Tensor of (batch_size,max_sentence_length)]
        '''
        tags_idx = [[self.tag_to_index[tag] for tag in sent] for sent in tags]
        tags_idx_padded, _ = self.pad_sents(tags_idx, Const.PAD_TAG_ID)
        tags_idx_padded = tags_idx_padded.to(device)
        return tags_idx_padded

    def ids_to_tags(self, ids):
        return [[self.idx_to_tag[x] for x in sent] for sent in ids]
