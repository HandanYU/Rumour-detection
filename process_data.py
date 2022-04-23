import pandas as pd
from sklearn import preprocessing
print('read dataset')
stat_feat = ['reply_like_count', 'reply_retweet_count',
       'reply_possibly_sensitive', 'reply_has_url', 'reply_mentioned_url_num',
       'reply_id_num', 'reply_isweekday', 'reply_senti_score', 'reply_count',
       'like_count', 'retweet_count', 'possibly_sensitive', 'has_url',
       'mentioned_url_num', 'id_num', 'isweekday', 'followers_count',
       'tweet_count', 'verified', 'senti_score']
dev_stat_feat_df = pd.read_csv('./data/filtered data/dev_stat_feat_df.csv')
train_stat_feat_df = pd.read_csv('./data/filtered data/train_stat_feat_df.csv')
test_stat_feat_df = pd.read_csv('./data/filtered data/test_stat_feat_df.csv')
covid_stat_feat_df = pd.read_csv('/Volumes/HANDAN/covid_stat_feat_df.csv')


print('process minmax.=========')     
minmax = preprocessing.MinMaxScaler()
train_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=train_stat_feat_df.index,
                                    data=minmax.fit_transform(train_stat_feat_df[stat_feat]))
dev_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=dev_stat_feat_df.index,
                                data=minmax.transform(dev_stat_feat_df[stat_feat]))
test_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=test_stat_feat_df.index,
                                data=minmax.transform(test_stat_feat_df[stat_feat]))
covid_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=covid_stat_feat_df.index,
                                data=minmax.transform(covid_stat_feat_df[stat_feat]))
dev_scaled_stat_feat_df.to_csv('./data/filtered data/dev_scaled_stat_feat_df.csv')
train_scaled_stat_feat_df.to_csv('./data/filtered data/train_scaled_stat_feat_df.csv')
test_scaled_stat_feat_df.to_csv('./data/filtered data/test_scaled_stat_feat_df.csv')
covid_scaled_stat_feat_df.to_csv('/Volumes/HANDAN/covid_scaled_stat_feat_df.csv')