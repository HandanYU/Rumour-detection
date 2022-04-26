import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from scipy import stats
print('read dataset')
stat_feat = ['reply_' + i for i in [ 'contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'isweekday','contributors_enabled', 'is_translator', 'is_translation_enabled',
       'has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']] + ['contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'isweekday', 'reply_count','contributors_enabled', 'is_translator', 'is_translation_enabled','has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']
dev_stat_feat_df = pd.read_csv('./tweepy_data/res/dev_stat_feat_df.csv')
train_stat_feat_df = pd.read_csv('./tweepy_data/res/train_stat_feat_df.csv')
test_stat_feat_df = pd.read_csv('./tweepy_data/res/test_stat_feat_df.csv')
dev_tweet_df = pd.read_csv('./tweepy_data/res/dev_tweet_df.csv')
train_tweet_df = pd.read_csv('./tweepy_data/res/train_tweet_df.csv')
# covid_stat_feat_df = pd.read_csv('/Volumes/HANDAN/covid_stat_feat_df.csv')
train_id = train_stat_feat_df['tweet_id']
dev_id = dev_stat_feat_df['tweet_id']
test_id = test_stat_feat_df['tweet_id']
train_label = train_stat_feat_df[['tweet_id', 'label']]
dev_label = dev_stat_feat_df[['tweet_id', 'label']]
nonan_stat_feat_df = []
tweet_df_ls = []
real_cols = train_stat_feat_df[stat_feat].dropna(axis=1, how='all').columns
for _, cur_df in train_stat_feat_df.groupby('label'):
       tweet_df_ls.extend(cur_df['tweet_id'].values)
       cur_df = cur_df[real_cols]
       imputer = KNNImputer(n_neighbors=5)
       print(cur_df.shape, imputer.fit_transform(cur_df).shape)
       nonan_stat_feat_df.append(pd.DataFrame(columns=real_cols, data=imputer.fit_transform(cur_df)))
train_stat_feat_df = pd.concat(nonan_stat_feat_df)
train_stat_feat_df['tweet_id'] = tweet_df_ls
train_stat_feat_df = train_stat_feat_df.drop_duplicates('tweet_id')
train_stat_feat_df = pd.merge(train_stat_feat_df, train_label, on='tweet_id', how='right')



nonan_stat_feat_df = []
tweet_df_ls = []
for _, cur_df in dev_stat_feat_df.groupby('label'):
       tweet_df_ls.extend(cur_df['tweet_id'].values)
       cur_df = cur_df[real_cols]
       imputer = KNNImputer(n_neighbors=5)
       nonan_stat_feat_df.append(pd.DataFrame(columns=real_cols, data=imputer.fit_transform(cur_df)))
dev_stat_feat_df = pd.concat(nonan_stat_feat_df)
dev_stat_feat_df['tweet_id'] = tweet_df_ls
dev_stat_feat_df = dev_stat_feat_df.drop_duplicates('tweet_id')
dev_stat_feat_df = pd.merge(dev_stat_feat_df, dev_label, on='tweet_id', how='right')

cur_df = test_stat_feat_df[real_cols]
imputer = KNNImputer(n_neighbors=5)
test_stat_feat_df = pd.DataFrame(columns=real_cols, data=imputer.fit_transform(cur_df))
test_stat_feat_df.index = test_id
# test_stat_feat_df = test_stat_feat_df.loc[test_id]
dev_stat_feat_df.to_csv('./tweepy_data/res/dev_stat_feat_nonan_df.csv')
train_stat_feat_df.to_csv('./tweepy_data/res/train_stat_feat_nonan_df.csv')
test_stat_feat_df.to_csv('./tweepy_data/res/test_stat_feat_nonan_df.csv')

print('process minmax.=========')     
minmax = preprocessing.StandardScaler()
train_scaled_stat_feat_df = pd.DataFrame(columns=real_cols, index=train_stat_feat_df['tweet_id'].values,
                                    data=minmax.fit_transform(train_stat_feat_df[real_cols]))
train_scaled_stat_feat_df['label'] = train_stat_feat_df['label'].values
dev_scaled_stat_feat_df = pd.DataFrame(columns=real_cols, index=dev_stat_feat_df['tweet_id'].values,
                                data=minmax.transform(dev_stat_feat_df[real_cols]))
dev_scaled_stat_feat_df['label'] = dev_stat_feat_df['label'].values
test_scaled_stat_feat_df = pd.DataFrame(columns=real_cols, index=test_stat_feat_df.index,
                                data=minmax.transform(test_stat_feat_df[real_cols]))
# covid_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=covid_stat_feat_df.index,
                            #     data=minmax.transform(covid_stat_feat_df[stat_feat]))
dev_scaled_stat_feat_df.to_csv('./tweepy_data/res/dev_scaled_stat_feat_df.csv')
train_scaled_stat_feat_df.to_csv('./tweepy_data/res/train_scaled_stat_feat_df.csv')
test_scaled_stat_feat_df.to_csv('./tweepy_data/res/test_scaled_stat_feat_df.csv')
# covid_scaled_stat_feat_df.to_csv('/Volumes/HANDAN/covid_scaled_stat_feat_df.csv')