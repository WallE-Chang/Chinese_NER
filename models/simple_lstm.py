# -*- coding: utf-8 -*-
# @Author: wanli
# @Date:   2019-07-24 18:19:06
# @Last Modified by:   wanli
# @Last Modified time: 2019-08-08 16:46:48
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class SimpleLSTM(nn.Module):

    def __init__(self, vocab_size, nb_labels, padding_idx=None,emb_dim=10, hidden_dim=10,pretrained_word_embedding=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if pretrained_word_embedding is not None:
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_word_embedding).float())
            assert emb_dim == pretrained_word_embedding.shape[1]
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim,padding_idx =padding_idx)

              
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, nb_labels)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden = None

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2),
            torch.randn(2, batch_size, self.hidden_dim // 2),
        )

    def forward(self, batch_of_sentences,mask):
        self.hidden = self.init_hidden(batch_of_sentences.shape[0])
        
        x = self.emb(batch_of_sentences)
        source_lengths = mask.sum(1)
        x, self.hidden = self.lstm(pack_padded_sequence(x, source_lengths,batch_first=True), self.hidden)
        x, input_sizes = pad_packed_sequence(x,batch_first=True )
        x = self.dropout(x)
        x = self.hidden2tag(x)
        return x

