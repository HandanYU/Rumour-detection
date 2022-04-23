import pandas as pd
import json
import numpy as np
from collections import defaultdict
from obtain_data import filter_feature, get_user_info
from process import clean_test_data, json2df, concat_user_info, concat_reply, check_weekday, concat_reply_info, extract_stat_tweet_feat
from textblob import TextBlob
# Concat json
data = []
for i in range(4):
    d = pd.read_json(path_or_buf=f'/Volumes/HANDAN/covid_source_data_{i}.jsonl', lines=True)
    data.append(d)
source_df = pd.concat(data)
print('finish merge source json')
reply_data = []
for i in range(4):
    d = pd.read_json(path_or_buf=f'/Volumes/HANDAN/covid_reply_data_{i}.jsonl', lines=True)
    reply_data.append(d)
reply_df = pd.concat(reply_data)
print('finish merge reply json')
source_feat_json = filter_feature(None, source_df)
print('filter source features')
reply_feat_json = filter_feature(None, reply_df)
print('filter reply features')
user_df = get_user_info(None, source_df)
print('get user info')
cleaned_source_df = clean_test_data(json2df(None, source_feat_json), False)
print('clean source')
cleaned_reply_df = clean_test_data(json2df(None, reply_feat_json), False)
print('clean reply')
cleaned_source_df = concat_user_info('covid', cleaned_source_df, user_df)
print('concat user info')
cleaned_source_df = concat_reply('covid', cleaned_source_df)
print('concat reply')
cleaned_source_df = check_weekday(cleaned_source_df)
print('check weekday')
cleaned_reply_df = check_weekday(cleaned_reply_df)

cleaned_source_df['senti_score'] = cleaned_source_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
cleaned_reply_df['senti_score'] = cleaned_reply_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
print('get senti_score')
cleaned_reply_df.index = cleaned_reply_df['tweet_id']
cleaned_source_df.index = cleaned_source_df['tweet_id']

cleaned_source_df.to_csv('./data/filtered data/cleaned_covid_df.csv')
print('get reply stat feat')
statis_feature = ['reply_count', 'like_count', 'retweet_count',
                        'possibly_sensitive', 'has_url', 'mentioned_url_num', 
                        'id_num', 'isweekday', 'senti_score']
df_iterator=pd.read_csv('./data/filtered data/cleaned_covid_df.csv')
with open('./data/original_data/covid_reply.txt', 'r') as f:
    content = f.readlines()
data_txt = np.loadtxt('./data/original_data/covid_reply.txt',dtype=str, delimiter=',')
data_txtDF = pd.DataFrame(columns=['tweet_id', 'source_id'], data=data_txt)
df_iterator.index = [str(i) for i in df_iterator['tweet_id']]
cleaned_reply_df.index = list(range(len(cleaned_reply_df)))
print('merge source id')
cleaned_reply_df_with_source_id = pd.merge(cleaned_reply_df, data_txtDF, on='tweet_id', how='left')
cleaned_reply_df_with_source_id.index = cleaned_reply_df_with_source_id['tweet_id']
stat_data = []
print('compute reply stat feat')
for source_id, df in cleaned_reply_df_with_source_id.groupby('source_id'):
    if source_id not in df_iterator.index:
        continue
    ids = [str(i).strip() for i in df_iterator.loc[source_id]['reply'].split(',')]
    cur_data = [source_id, ' [SEP] '.join(df.loc[ids]['text'].values) + ' [SEP]']
    for i in statis_feature:
        cur_data.append(df[i].sum())
    stat_data.append(cur_data)
reply_stat_df = pd.DataFrame(columns=['tweet_id', 'reply_text'] + ['reply_' + s for s in statis_feature], data=stat_data)
cleaned_source_df.index = list(range(len(cleaned_source_df)))
covid_df = pd.merge(cleaned_source_df, reply_stat_df, on='tweet_id', how='left')
covid_stat_feat_df, covid_tweet_df = extract_stat_tweet_feat(False, covid_df)
covid_tweet_df.to_csv('/Volumes/HANDAN/covid_tweet_df.csv')
covid_stat_feat_df.to_csv('/Volumes/HANDAN/covid_stat_feat_df.csv')
