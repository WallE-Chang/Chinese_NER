# -*- coding: utf-8 -*-
# @Author: wanli
# @Date:   2019-07-24 18:19:50
# @Last Modified by:   wanli
# @Last Modified time: 2019-08-08 16:28:37

import sys
sys.path.append("./models")

from config import Const
from crf_vectorized import CRF

from simple_lstm import SimpleLSTM
from torch import nn


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, nb_labels, padding_idx,emb_dim=5, hidden_dim=4,pretrained_word_embedding=None):
        super().__init__()
        self.lstm = SimpleLSTM(
            vocab_size, nb_labels,padding_idx=padding_idx, emb_dim=emb_dim, hidden_dim=hidden_dim,pretrained_word_embedding=pretrained_word_embedding
        )
        self.crf = CRF(
            nb_labels,
            Const.BOS_TAG_ID,
            Const.EOS_TAG_ID,
            pad_tag_id=Const.PAD_TAG_ID,  # try setting pad_tag_id to None
            batch_first=True,
        )

    def forward(self, x, mask=None):
        emissions = self.lstm(x,mask)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x, y, mask=None):
        emissions = self.lstm(x,mask)
        nll = self.crf(emissions, y, mask=mask)
        return nll
