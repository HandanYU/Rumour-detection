import pandas as pd
from sklearn import preprocessing
print('read dataset')
stat_feat = ['reply_' + i for i in [ 'contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday','contributors_enabled', 'is_translator', 'is_translation_enabled',
       'has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']] + ['contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday', 'reply_count','contributors_enabled', 'is_translator', 'is_translation_enabled','has_extended_profile', 'default_profile', 'default_profile_image', 'following', 'follow_request_sent', 'notifications']
dev_stat_feat_df = pd.read_csv('./tweepy_data/res/dev_stat_feat_df.csv')
train_stat_feat_df = pd.read_csv('./tweepy_data/res/train_stat_feat_df.csv')
test_stat_feat_df = pd.read_csv('./tweepy_data/res/test_stat_feat_df.csv')
# covid_stat_feat_df = pd.read_csv('/Volumes/HANDAN/covid_stat_feat_df.csv')
nonan_stat_feat_df = []
for _, cur_df in train_stat_feat_df.groupby('label'):
    nonan_stat_feat_df.append(cur_df.fillna(cur_df.mean()))
train_stat_feat_df = pd.concat(nonan_stat_feat_df)
nonan_stat_feat_df = []
for _, cur_df in dev_stat_feat_df.groupby('label'):
    nonan_stat_feat_df.append(cur_df.fillna(cur_df.mean()))
dev_stat_feat_df = pd.concat(nonan_stat_feat_df)
test_stat_feat_df = test_stat_feat_df.fillna(test_stat_feat_df.mean())
print('process minmax.=========')     
minmax = preprocessing.StandardScaler()
train_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=train_stat_feat_df.index,
                                    data=minmax.fit_transform(train_stat_feat_df[stat_feat]))
dev_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=dev_stat_feat_df.index,
                                data=minmax.transform(dev_stat_feat_df[stat_feat]))
test_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=test_stat_feat_df.index,
                                data=minmax.transform(test_stat_feat_df[stat_feat]))
# covid_scaled_stat_feat_df = pd.DataFrame(columns=stat_feat, index=covid_stat_feat_df.index,
                            #     data=minmax.transform(covid_stat_feat_df[stat_feat]))
dev_scaled_stat_feat_df.to_csv('./tweepy_data/res/dev_scaled_stat_feat_df.csv')
train_scaled_stat_feat_df.to_csv('./tweepy_data/res/train_scaled_stat_feat_df.csv')
test_scaled_stat_feat_df.to_csv('./tweepy_data/res/test_scaled_stat_feat_df.csv')
# covid_scaled_stat_feat_df.to_csv('/Volumes/HANDAN/covid_scaled_stat_feat_df.csv')