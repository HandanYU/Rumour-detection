import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import BertModel
import torch.optim as optim
import time
class TweetDataset(Dataset):
    def __init__(self, data_type, max_seq_len):
        self.max_seq_len = max_seq_len
        # read pre-processed data
        self.tweet_df = pd.read_csv(f'./data/filtered data/{data_type}_tweet_df.csv', usecols=['text', 'reply_text', 'label'])
        self.statistic_df = pd.read_csv(f'./data/filtered data/{data_type}_stat_feat_df.csv').drop(columns='tweet_id')
        self.tweet_df['text'] = self.tweet_df['text'].replace(np.nan, '')
        self.tweet_df['reply_text'] = self.tweet_df['reply_text'].replace(np.nan, '')
        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # DistilBertTokenizerFast.from_pretrained("bert-base-uncased")
    def __len__(self):
        return self.tweet_df.shape[0]
    def __getitem__(self, idx):
        # source_token_mask = self.tokenizer(self.tweet_df.iloc[idx]['text'], truncation=True, padding='max_length', max_length=self.max_seq_len)
        # source_token, source_mask = torch.tensor(source_token_mask['input_ids']), torch.tensor(source_token_mask['attention_mask'])
        concat_text = '[CLS] '+ self.tweet_df.iloc[idx]['text'] + ' [SEP] ' + self.tweet_df.iloc[idx]['reply_text']
        concat_tokens = self.tokenizer.tokenize(concat_text)
        if len(concat_tokens) < self.max_seq_len:
          concat_tokens_padded = concat_tokens + ['[PAD]' for _ in range(self.max_seq_len - len(concat_tokens))]
        else:
          concat_tokens_padded = concat_tokens[:self.max_seq_len-1] + ['[SEP]']
        concat_token_ids = self.tokenizer.convert_tokens_to_ids(concat_tokens_padded)
        concat_attn_masks = [1 if token != '[PAD]' else 0 for token in concat_tokens_padded]
        concat_seg_ids = []
        seg_idx = 0
        for i in range(len(concat_token_ids)):
            concat_seg_ids.append(seg_idx)
            if concat_tokens_padded[i] == '[SEP]':
                seg_idx += 1
        concat_token_ids_tensor = torch.tensor(concat_token_ids)
        concat_attn_masks_tensor = torch.tensor(concat_attn_masks)
        concat_seg_ids_tensor = torch.tensor(concat_seg_ids)
        # pair_token_mask = self.tokenizer(self.tweet_df.iloc[idx]['text'], self.tweet_df.iloc[idx]['reply_text'], truncation='only_second', padding='max_length', max_length=self.max_seq_len)
        # pair_tokens_tensor, pair_mask_tensor = torch.tensor(pair_token_mask['input_ids']), torch.tensor(pair_token_mask['attention_mask'])
        return concat_token_ids_tensor, concat_attn_masks_tensor, concat_seg_ids_tensor, self.tweet_df.iloc[idx]['label'], torch.Tensor(self.statistic_df.iloc[idx])

class RumorClassifier(nn.Module):

    def __init__(self):
        super(RumorClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        self.ffnn = nn.Sequential(nn.Linear(791,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, seq, attn_masks, seg, stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask = attn_masks, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        
        x = torch.cat((cls_rep,stats),dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(x)

        return logits
def get_accuracy_from_logits(logits, labels):
    probs = logits.unsqueeze(-1)
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, criterion, dataloader, device):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            logits = net(seq)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count
if __name__ == '__main__':
  train_loader = DataLoader(TweetDataset('train', 256), shuffle=True, batch_size=64, drop_last=True)
  dev_loader = DataLoader(TweetDataset('dev', 256), shuffle=True, batch_size=64, drop_last=True)
  torch.cuda.empty_cache ()
  net = RumorClassifier()
  # net = net.to(device)
  
  criterion = nn.BCELoss()
  opti = optim.Adam(net.parameters(), lr = 2e-5)

  best_acc = 0
  st = time.time()
  eps = []
  t_loss = []
  d_loss = []
  for ep in range(5):
    eps.append(ep)
    net.train()
  
    for it, (seq, mask, seg, labels,stats) in enumerate(train_loader):
        
        #Clear gradients
        opti.zero_grad()
        #Converting these to cuda tensors
        # seq, mask, seg, labels = seq.to(device), mask.to(device), seg.to(device), labels.to(device)
        
        #Obtaining the logits from the model
        print(stats.shape)
        logits = net(seq, mask, seg, stats)
        
        #Computing loss
        loss = criterion(logits.squeeze(), labels.float())

        #Backpropagating the gradients
        loss.backward()

        #Optimization step
        opti.step()

        if it % 10 == 0:

            acc = get_accuracy_from_logits(logits, labels)
            print("Iteration {} of epoch {} complete. \n Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, loss.item(), acc, (time.time()-st)))
            st = time.time()

        
    dev_acc, dev_loss = evaluate(net, criterion, dev_loader, 'cpu')
    t_loss.append(loss.item())
    d_loss.append(dev_loss)
    print("Development Accuracy: {}; Development Loss: {}".format(dev_acc, dev_loss))
    if dev_acc > best_acc:
        # print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
        best_acc = dev_acc
        torch.save(net.state_dict(), 'bertcls_{}.dat'.format(ep))
