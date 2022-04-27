import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from transformers import BertModel
from transformers import BertTokenizer
import time
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score
LOG_FOUT = open(os.path.join('./log', 'log_lstm.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
def get_f1_from_logits(logits, labels):
    preds = (logits > 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    return f
def get_roc_auc_from_logits(logits, labels):
    preds = (logits > 0.5).astype(int)
    return roc_auc_score(preds,labels)
class TweetDataset(Data.Dataset):
    def __init__(self, data_type, max_len):
        tweet_df = pd.read_csv(f'{data_type}_tweet_df.csv')
        static_df = pd.read_csv(f'{data_type}_scaled_stat_feat_df.csv')
        static_df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
        zero_cols = []
        for column in static_df.columns:
            if static_df[column].sum() == 0:
                zero_cols.append(column)
        static_df.drop(columns=zero_cols, inplace=True)
        static_df.fillna('', inplace=True)
        tweet_df.reply_text.fillna('', inplace=True)
        tweet_df.text.fillna('', inplace=True)
        tweet_df.text = tweet_df.apply(lambda x: '[CLS] ' + str(x['text']).strip() + ' [SEP] ' + str(x['reply_text']).strip() + ' [SEP]', axis=1)
        self.tweet_df = tweet_df
        self.static_df = static_df
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self):
        return len(self.tweet_df)
    def __getitem__(self, idx):
        txt = self.tweet_df.text.iloc[idx]
        static = self.static_df.iloc[idx]
        label = self.tweet_df.label.iloc[idx]
        tokens = self.tokenizer.tokenize(txt)
        if len(tokens) < self.max_len:
            padded_tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            padded_tokens = tokens[: self.max_len - 1] + ['[SEP]']
        attn_mask = [0 if i == '[PAD]' else 1 for i in padded_tokens]
        seg_index = 0
        seg_id = []
        for i in padded_tokens:
            seg_id.append(seg_index)
            if i == '[SEP]':
                if seg_index == 1:
                    seg_index = 0
                else:
                    seg_index = 1
        token_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)

        # print( torch.Tensor(token_ids).shape,
        #  torch.Tensor(attn_mask).shape,
        #   torch.Tensor(seg_id).shape, torch.Tensor(static).shape, torch.LongTensor(label).shape)
        return torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(seg_id), torch.Tensor(static), torch.tensor(label), idx
        # torch.tensor(static), torch.LongTensor(label), idx



class LSTMClassifier(nn.Module):
    """
    LSTMClassifier classification model
    """
    def __init__(self, dropout_prob=0.3):
        super().__init__()
        self.embedding_layer = BertModel.from_pretrained('bert-base-uncased')
        """
           Define the components of your BiLSTM Classifier model
           2. TODO: Your code here
        """
        self.embed_dim = 768 # embeddings.shape[1] # 300-dim of Glove embeddings
        self.hidden_size = 256
        self.num_classes = 1
        self.num_layers = 2
        self.directions = 2

        self.dropout = nn.Dropout(p=dropout_prob) 
        self.lstm = nn.LSTM(
            768, self.hidden_size, self.num_layers,
            dropout=dropout_prob, bidirectional=True,
            batch_first=True) 
        self.non_linearity =nn.ReLU() # For example, ReLU
#         self.lr = nn.Sequential(nn.Linear(43,32),
#                           nn.ReLU(),
#                           nn.Dropout(0.3),
#                          nn.Linear(32,1),
#                           nn.Sigmoid())
        self.clf = nn.Sequential(nn.Linear(self.hidden_size*self.directions + 43, self.num_classes) , nn.Sigmoid())
    


    def forward(self, seq_tokens, attn_mask, seg, stat):
        logits = None
        """
           Write forward pass for LSTM. You must use dropout after embedding
           the inputs. 

           Example, forward := embedding -> bilstm -> pooling (sum?mean?max?) 
                              nonlinearity -> classifier
           Refer to: https://arxiv.org/abs/1705.02364

           Return logits

           3. TODO: Your code here
        """
        out = self.embedding_layer(seq_tokens, attn_mask, token_type_ids=seg).last_hidden_state[:, 0] #
        inputs_dropout = self.dropout(out) 
        
        inputs_dropout = inputs_dropout.unsqueeze(1)
        lstm_out, (lstm_ht, _) = self.lstm(inputs_dropout, None) 
        outs = torch.cat([lstm_out[:, -1, :].squeeze(1), stat], dim=1)
        nonlinear_out = self.non_linearity(outs) 
        logits = self.clf(nonlinear_out)
#         stat_logit = self.ln(stat)
        return logits
    
    
        
        
class RumorClassifier(nn.Module):

    def __init__(self):
        super(RumorClassifier, self).__init__()
        #Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        self.lr = nn.Sequential(nn.Linear(811,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())

    def forward(self, seq, attn_masks, seg, stat):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=seg, return_dict=True)
        cont_reps = outputs.last_hidden_state
        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        inputs = torch.cat([cls_rep, stat], dim=1)
        out = self.lr(inputs)
        # #Feeding cls_rep to the classifier layer
        # embs = self.ffnn(x)

        return out
def evaluate(net, criterion, dataloader, device):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0
    all_log = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for seq, mask, seg, stats, labels, idx in dataloader:
            labels = labels.to(device)
            #Obtaining the logits from the model
            logits = net(seq.to(device), mask.to(device), seg.to(device), stats.to(device))
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            # mean_acc += get_accuracy_from_logits(logits, labels)

            all_log = np.hstack((all_log, logits.squeeze().cpu().numpy()))
            all_labels = np.hstack((all_labels, labels.cpu().numpy()))
            count += 1

        f = get_f1_from_logits(all_log, all_labels)
        roc_auc = get_roc_auc_from_logits(all_log, all_labels)
    return f, roc_auc, mean_loss / count
print('load data.')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
train_data = pd.read_csv('train_tweet_df.csv')

# y_train = train_data['label']

# w_nonr = len(y_train)/(len(y_train)-y_train.sum())
# w_r = len(y_train)/(y_train.sum())
# weights = []
# for l in y_train:
#     if l == 0:
#         weights.append(w_nonr)
#     else:
#         weights.append(w_r)
# weights = torch.FloatTensor(weights)
# train_stat = pd.read_csv('train_scaled_stat_feat_df.csv')
class_counts = torch.tensor([len(train_data)-train_data.label.sum(), train_data.label.sum()])
weight = 1 / class_counts
weights = torch.FloatTensor([weight[i] for i in train_data.label])
train_set = TweetDataset('train', 256)
dev_set = TweetDataset('dev', 256)
train_sampler = Data.WeightedRandomSampler(weights, len(train_set), replacement=True)
train_loader = Data.DataLoader(train_set, sampler=train_sampler, batch_size=64)
dev_loader = Data.DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('load model.')
net = LSTMClassifier()
net.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), 2e-5)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

st = time.time()
print('begin training.')
for ep in range(20):
    net.train()
    whole_loss = 0
    for it, (seq, mask, seg, train_stat, labels, idx) in enumerate(train_loader):
        seq, mask, seg, train_stat, labels, idx = seq.to(device), mask.to(device), seg.to(device), train_stat.to(device), labels.to(device), idx.to(device)
        #Clear gradients
        optimizer.zero_grad()
        #Converting these to cuda tensors
        # seq, mask, seg, labels = seq.to(device), mask.to(device), seg.to(device), labels.to(device)

        #Obtaining the logits from the model
        logits = net(seq, mask, seg, train_stat)
        #Computing loss
        loss_1 = criterion(logits.squeeze(1), labels.float())
#         loss_2 = criterion(stat_logit.squeeze(1), labels.float())
#         avg_loss = (loss_1.item() + loss_2.item()) / 2
        whole_loss += loss_1
        #Backpropagating the gradients
        loss_1.backward()
#         loss_2.backward()

        #Optimization step
        optimizer.step()
        
        if it % 10 == 0:

            log_string("Iteration {} of epoch {} complete. Time taken (s): {}".format(it, ep, (time.time()-st)))
            st = time.time()
#     scheduler.step()
    log_string(f'epoch: {ep}, avg_loss: {whole_loss / (it+1)}')
    dev_acc, roc_auc, dev_loss = evaluate(net, criterion, dev_loader, device)
#     net.eval()
#     with torch.no_grad():
#         for seq, mask, seg, stat, labels, idx in dev_loader:
#             seq, mask, seg, stat, labels, idx = seq.to(device), mask.to(device), seg.to(device), stat.to(device), labels.to(device), idx.to(device)
#             dev_embs = net(seq, mask, seg, stat)
#         dev_loss = criterion(dev_embs.squeeze(1), labels.float())
# #         lr_loss = criterion(lr_logit.squeeze(1), labels.float())
# #         avg_dev_loss = (dev_loss.item() + lr_loss.item()) / 2
#         f1 = get_f1_from_logits(dev_embs.squeeze(1).cpu().numpy(), labels.cpu().numpy())
        
#         roc = get_roc_auc_from_logits(dev_embs.squeeze(1).cpu().numpy(), labels.cpu().numpy())
    log_string("Development F1: {}; Development ROCAUC: {}; Development Loss: {}".format(dev_acc, roc_auc, dev_loss))
    torch.save({
                  'model': net.state_dict(),
                #   'optimizer': opti.state_dict(),
                #   'scheduler': scheduler.state_dict(),
                #   'iteration': ep
                }, f'./models/bertemb_{ep}.dat')
