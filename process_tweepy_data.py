import os
import tqdm
import json
import pandas as pd
from textblob import TextBlob
import pandas as pd
from sklearn import preprocessing
from process import clean_test_data, concat_reply, check_weekday_test, concat_reply_info, extract_stat_tweet_feat
def merge_json(data_type, source_or_reply, ids_list):
    # merged_json: "test_source.json"
    merges_file = os.path.join(f'./tweepy_data/objects/', f'{data_type}_{source_or_reply}.json')
    path_results = f'./tweepy_data/objects/{data_type}_objects'
    with open(merges_file, "w", encoding="utf-8") as f0:
        for file in os.listdir(path_results):
            if file.split('.')[0] in ids_list:
                print('write')
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    for line in tqdm.tqdm(f1):
                        line_dict = json.loads(line)
                        js = json.dumps(line_dict, ensure_ascii=False)
                        f0.write(js + '\n')
                    f1.close()
        f0.close()

def sort_by_time(raw_file, json_file):
    with open(raw_file) as file:
        ids = file.readlines()
    df = pd.read_json(path_or_buf=json_file, lines=True)
    df.index = [str(i) for i in df['id']]
    save_name = raw_file[:-4] + '_sorted.txt'
    with open(save_name, 'w') as file:
        date = pd.Series(pd.DatetimeIndex(df['created_at']), index=df.index)
        df.drop(['created_at'], axis=1, inplace=True)
        df['time'] = date
        for id_ in ids:
            ids_ = id_.strip().split(',')
            source_id = ids_[0]
            file.write(source_id)
            if len(ids_) > 1:
                reply_ids = ids_[1:]
                reply_ids[-1] = reply_ids[-1].strip()
                valid_ids = [index for index in reply_ids if index in df.index]
                sorted_replies = df.loc[valid_ids].sort_values(by='time')
                if len(valid_ids) > 0:
                    file.write(',')

                for i, index in enumerate(sorted_replies.index):
                    file.write(index)
                    if i != len(sorted_replies.index) - 1:
                        file.write(',')

            file.write('\n')
def concat_reply(data_type, source_df):
    """concat replies on source tweets
    data_type: 'dev', 'train', 'test'
    """
    df = pd.DataFrame(columns=['tweet_id', 'reply'])
    with open(f'./tweepy_data/original_data/{data_type}.data_sorted.txt', 'r') as f:
    # with open(f'./data/original_data/{data_type}.data_sorted.txt', 'r') as f:
        content = f.readlines()
    df['tweet_id'] = [c.split(',')[0].strip() for c in content]
    df['reply'] = [','.join([i.strip() for i in c.split(',')[1:]]) for c in content]
    source_df = pd.merge(source_df, df, on='tweet_id', how='left')
    return source_df
def concat_label(data_type, source_feature_df):
    """Concat labels on source tweets
    data_type: 'dev', 'train'
    """
    df = pd.DataFrame(columns=['tweet_id', 'label'])
    with open(f'./tweepy_data/original_data/{data_type}_source.txt', 'r') as f:
        ids = f.readlines()
    with open(f'./tweepy_data/original_data/{data_type}.label.txt', 'r') as f:
        labels = f.readlines()
    df['tweet_id'] = [id.strip() for id in ids]
    df['label'] = [label.strip() for label in labels]
    df_labels = pd.merge(source_feature_df, df, on='tweet_id', how='left')
    df_labels['label'] = df_labels['label'].apply(lambda x: 0 if x == 'nonrumour' else 1)
    return df_labels


def processing(data_type):

    # './data/tweet-objects/test_source.json'
    used_cols = ['created_at', 'id', 'id_str', 'text', 'truncated', 'entities', 'source',
       'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place',
       'contributors', 'is_quote_status', 'retweet_count', 'favorite_count',
       'favorited', 'retweeted', 'lang', 'extended_entities',
       'possibly_sensitive', 'possibly_sensitive_appealable',
       'quoted_status_id', 'quoted_status_id_str', 'quoted_status']
    source_df = pd.read_json(path_or_buf=f'./tweepy_data/objects/{data_type}_source.json', lines=True)
    source_df = source_df[used_cols]
    reply_df= pd.read_json(path_or_buf=f'./tweepy_data/objects/{data_type}_reply.json', lines=True)
    reply_df = reply_df[used_cols]
    source_df = clean_test_data(source_df)
    reply_df = clean_test_data(reply_df)
    
    
    # get 'verified', 'followers_count', 'listed_count'
    for i in [ 'protected', 'followers_count', 'friends_count', 
                'listed_count', 'favourites_count', 'geo_enabled', 'verified', 
                'statuses_count', 'contributors_enabled']:
        source_df[i] = source_df['user'].apply(lambda x: x[i])
        reply_df[i] = reply_df['user'].apply(lambda x: x[i])
    source_df['has_url'] = source_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    # get reply statistic info
    reply_df['has_url'] = reply_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    source_df = concat_reply(data_type, source_df)
    # get reply count
    source_df['reply_count'] = source_df['reply'].apply(lambda x: len(x.split(',')))
    source_df.index = [str(i) for i in source_df['tweet_id']]

    # sorted ids
    if data_type == 'test':
        with open('tweep_data/original_data/test_source.txt', 'r') as f:
            c = f.readlines()
        source_df = source_df.loc[[i.strip() for i in c]]
    source_df = check_weekday_test(source_df)
    reply_df = check_weekday_test(reply_df)
    # add sentiment score
    source_df['senti_score'] = source_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df['senti_score'] = reply_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df.index = [str(i) for i in reply_df['tweet_id']]
    # reply_df = reply_df.rename(columns={'retweet_count': 'retweet_count', 'favorite_count': 'like_count',
    #                                     'mentioned_url_num': 'mentioned_url_num', 'id_num': 'id_num', 'isweekday': 'isweekday'})
    # source_df.index = source_df['tweet_id']
    # source_df = source_df.rename(columns={'retweet_count': 'retweet_count', 'favorite_count': 'like_count', 'followers_count': 'followers_count',
    #                                     'mentioned_url_num': 'mentioned_url_num', 'id_num': 'id_num', 'isweekday': 'isweekday', 
    #                                     'verified': 'verified', 'listed_count': 'tweet_count'})
    # concat replies info to source_df
    # reply_count, quote_count
    # count_feat = ['in_reply_to_status_id',
    #    'in_reply_to_user_id','quoted_status_id']
    # for c in count_feat:
    #     source_df[c] = source_df[c].apply(lambda x: 1 if x)
    statis_feature = [ 'contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable', 'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday']
    source_df[['reply_text'] + ['reply_' + s for s in statis_feature]] = source_df.apply(lambda x: concat_reply_info(x['reply'], reply_df, statis_feature), axis=1, result_type='expand')          
    if data_type == 'train':
        source_df = concat_label(data_type, source_df)
    # source_df[['reply_reply_count', 'reply_quote_count', 'quote_count']] = 0      
    return source_df

def split_source_reply(txt_file):
  """
  txt_file: 'train.data.txt'
  """
  with open(txt_file) as f:
      ids = f.readlines()
  source_ids = []
  reply_ids = []
  source_txt_file = txt_file.split('.')[0] + '_source.txt'
  reply_txt_file = txt_file.split('.')[0] + '_reply.txt'
  for i in range(len(ids)):
      source_ids.append(ids[i].split(',')[0].strip())
      reply_ids.extend([r.strip() for r in ids[i].split(',')[1:]])
# save source_ids
  with open(source_txt_file,'w') as f:
      for i in source_ids:
          f.write(i)
          f.write('\n')
# save reply_ids
  with open(reply_txt_file,'w') as f:
      for i in reply_ids:
        f.write(i)
        f.write('\n')
def extract_stat_tweet_feat(istrain, df):
    # extract statistic features
    # reply_reply_count， reply_quote_count，quote_count
    statistic_features = ['reply_' + i for i in ['contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
       'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday']] + [
        'contributors',
       'possibly_sensitive', 'possibly_sensitive_appealable',
        'retweet_count', 'favorite_count', 'mentioned_url_num', 'id_num',
       'followers_count', 'friends_count', 'listed_count', 'favourites_count',
       'statuses_count', 'has_url', 'senti_score','truncated', 'is_quote_status', 'favorited', 'retweeted', 'protected',
       'geo_enabled', 'verified', 'contributors_enabled', 'isweekday', 'reply_count']
    stat_feat_df = df[statistic_features]
    stat_feat_df.index = df['tweet_id']
    if istrain:
        tweet_df = df[['tweet_id', 'text', 'reply_text', 'label']]
    else:
        tweet_df = df[['tweet_id', 'text', 'reply_text']]
    # tweet_df = df.drop(columns=statistic_features)
    tweet_df.index = df['tweet_id']
    # convert into float
    # for col in ['tweet_count', 'followers_count', 'verified']:
    #     stat_feat_df[col] = stat_feat_df[col].apply(lambda x: float(x))
    # fill nan using corresponding mean
    stat_feat_df = stat_feat_df.fillna(stat_feat_df.mean())
    return stat_feat_df, tweet_df
split_source_reply('tweepy_data/original_data/test.data.txt')
with open('tweepy_data/original_data/test_source.txt', 'r') as f:
    content = f.readlines()
source_ids = [c.strip() for c in content]
with open('data/original_data/test_reply.txt', 'r') as f:
    content = f.readlines()
reply_ids = [c.strip() for c in content]
merge_json('test', 'source', source_ids)
merge_json('test','reply', reply_ids)
raw_files = ['./tweepy_data/original_data/test.data.txt']
json_files = ['./tweepy_data/objects/test_reply.json']
for raw_file, json_file in zip(raw_files, json_files):
    sort_by_time(raw_file, json_file)
test_df = processing('test')
test_stat_feat_df, test_tweet_df = extract_stat_tweet_feat(True, test_df)