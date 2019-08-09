# -*- coding: utf-8 -*-
# @Author: wanli
# @Date:   2019-08-08 17:32:03
# @Last Modified by:   changwanli
# @Last Modified time: 2019-08-09 15:02:23

import numpy as np
import torch
import torch.optim as optim

from data_proprecess import data_preprocess
from models.bilstm_crf import BiLSTM_CRF
from utils import batch_iter
from utils import load_embedding
from utils import read_corpus

# In[1] define some parameters
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pretained_word_embedding = False


# In[2] load data
training_data = read_corpus('./dataset/train_data')
test_data = read_corpus('./dataset/test_data')

# In[3] define data proprecess
if use_pretained_word_embedding:
    word_to_index, index_to_vector = load_embedding(
        './Chinese_word_embedding/Tencent_AILab_ChineseEmbedding.txt')
    dp = data_preprocess(training_data, pretrained_word_to_index=word_to_index)
else:
    dp = data_preprocess(training_data, pretrained_word_to_index=None)

# In[4] define model
if use_pretained_word_embedding:
    model = BiLSTM_CRF(len(dp.word_to_index), len(dp.tag_to_index), padding_idx=dp.word_to_index[
                       "<pad>"], emb_dim=200, hidden_dim=50, pretrained_word_embedding=index_to_vector)
else:
    model = BiLSTM_CRF(len(dp.word_to_index), len(dp.tag_to_index), padding_idx=dp.word_to_index[
                       "<pad>"], emb_dim=200, hidden_dim=50, pretrained_word_embedding=None)
model = model.to(device)

# In[5] define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# In[6] train and test
for epoch in range(300):
    print("=" * 100)
    print(f'epoch :{epoch}')

    # train
    total_loss = 0
    iteration_num = 0
    model.train()
    for sentences, tags in batch_iter(training_data, 1024 * 512, shuffle=True):
        X, mask = dp.to_input_idx(sentences, device)
        y = dp.to_input_label(tags, device)
        loss = model.loss(X, y, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().detach().numpy()
        iteration_num += 1
    print('train loss :%.2f' % (total_loss / iteration_num))

    # test
    model.eval()
    total_loss = 0
    total_acc = 0
    iteration_num = 0
    with torch.no_grad():
        for sentences, tags in batch_iter(test_data, 1024 * 512):
            X, mask = dp.to_input_idx(sentences, device)
            y = dp.to_input_label(tags, device)
            loss = model.loss(X, y, mask=mask)
            scores, path = model(X, mask=mask)
            labels_flatten = np.concatenate(
                [[dp.tag_to_index[tag] for tag in sent] for sent in tags])
            path_flatten = np.concatenate(path)
            total_acc += np.mean(labels_flatten == path_flatten)
            total_loss += loss.cpu().detach().numpy()
            iteration_num += 1

    print('test loss :%.2f' % (total_loss / iteration_num))
    print('test acc :%.8f' % (total_acc / iteration_num))

# In[7] test your sentence
X, mask = dp.to_input_idx([list("邓小平喜欢吃桃子，可是毛泽东喜欢吃巧克力")], device)
scores, path = model(X, mask=mask)

print(dp.ids_to_tags(path))
