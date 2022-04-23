# -*- coding: utf-8 -*
import emoji
from nltk.corpus import stopwords
import re
import nltk
import json
import pandas as pd
from utils import timer
from datetime import datetime
from textblob import TextBlob
import numpy as np
from sklearn import preprocessing
from time import strftime
nltk.download('wordnet')
stemmer = nltk.stem.porter.PorterStemmer()
stopword = stopwords.words('english') 
def clean_tweet(content):
    

    # def compute_num_month(content):
    #     month_num = 0
    #     month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "octorber", "november", "december", 
    #             "jan.", "feb.", "mar.", "apr.", "may.", "jun.", "jul.", "aug.", "sept.", "oct.", "nov.", "dec."]
    #     for i in content:
    #         if i in month:
    #             month_num += 1
    #     return month_num
    # replace_abbreviations
    
    content = content.lower()
    content = re.sub(r"won't", "will not", content)
    content = re.sub(r"can't", "can not", content)
    content = re.sub(r"cannot", "can not", content)
    content = re.sub(r"n't", " not", content)
    content = re.sub(r"'re", " are", content)
    content = re.sub(r"'s", " is", content)
    content = re.sub(r"'d", " would", content)
    content = re.sub(r"'ll", " will", content)
    content = re.sub(r"'t", " not", content)
    content = re.sub(r"'ve", " have", content)
    content = re.sub(r"'m", " am", content)
    content = re.sub(r".”", " ", content)
    
    # get the number of month be mentioned
    # month_num = compute_num_month(content)

    # get the number of url
    mentioned_url_num = len(re.findall(r'https?://[^ ]+', content))
    mentioned_url_num += len(re.findall(r'www.[^ ]+', content))
    # get the number of twitter ID be mentioned
    id_num = len(re.findall(r'@[A-Za-z0-9_]+', content))
    content = re.sub(r'@[A-Za-z0-9_]+', '', content) # remove twitter ID
    # remove url 
    ## http, https
    content = re.sub(r'https?://[^ ]+', '', content) 
    ## www.
    content = re.sub(r'www.[^ ]+', '', content)
    # get the emoji and replace them as words
    # emojis = emoji.distinct_emoji_list(content)
    # for e in emojis:
    #     try:
    #         content = re.sub(e, emoji.demojize(e), content)
    #     except:
    #         print('no corresponding emoji!')
    #         print(content)
    #         content = re.sub(e, '', content)
    content = re.sub('\w+\d+\w+', '', content) # remove the word contains numbers
    
    content = re.sub(r'[:_!\+“\-=——,$%^\?\\~\"\'@#$%&\*<>{}\[\]()/]', ' ', content) # remove punctuation, except .
    
    content = re.sub(r"\s+", " ", content) # conver multiple spaces as a single space
    content = content.strip()
    
    # remove stop words 
    # TODO: and keep only english letters
    
    content = [c for c in content.split(' ') if c not in stopword and c.isalpha()]
    # do stemming
    content = [stemmer.stem(token.strip()) for token in content]
    
    return ' '.join(content), mentioned_url_num, id_num 
    #, month_num

# @timer('ms')
def json2df(json_file, json_content=None):
    """
    json_file: 'train_reply.json'
    """
    print(json_content is None)
    if json_file is not None:
        with open(json_file,'r+') as file:
            content = file.read()
    else:
        content = json_content
    content = json.loads(content)
    df = pd.DataFrame(content)
    df = df.T
    return df


# df = json2df('train_reply.json')
def clean_data(data_type):
  """
  Args: 
    data_type: 'dev', 'train'
  Returns:
    source_df, reply_df
  """
  source_df = json2df(f'./data/full data/{data_type}_source.json')
  source_df['temp'] = source_df['text'].apply(lambda x: clean_tweet(x))
  source_df['text'] = source_df['temp'].apply(lambda x: x[0])
  source_df['mentioned_url_num'] = source_df['temp'].apply(lambda x: x[1])
  source_df['id_num'] = source_df['temp'].apply(lambda x: x[2])
  source_df = source_df.drop(columns='temp')
  source_df['tweet_id'] = source_df.index

  reply_df = json2df(f'./data/full data/{data_type}_reply.json')
  reply_df['temp'] = reply_df['text'].apply(lambda x: clean_tweet(x))
  reply_df['text'] = reply_df['temp'].apply(lambda x: x[0])
  reply_df['mentioned_url_num'] = reply_df['temp'].apply(lambda x: x[1])
  reply_df['id_num'] = reply_df['temp'].apply(lambda x: x[2])
  reply_df = reply_df.drop(columns='temp')
  reply_df['tweet_id'] = reply_df.index
  
  return source_df, reply_df
# train_source_df, train_reply_df = clean_data('train')
# dev_source_df, dev_reply_df = clean_data('dev')
def clean_test_data(df, is_test=True):
    df['temp'] = df['text'].apply(lambda x: clean_tweet(x))
    df['text'] = df['temp'].apply(lambda x: x[0])
    df['mentioned_url_num'] = df['temp'].apply(lambda x: x[1])
    df['id_num'] = df['temp'].apply(lambda x: x[2])
    df['tweet_id'] = [str(i) for i in df.index]
    if is_test:
        df['tweet_id'] = df['id'].apply(lambda x: str(x))
        df = df.drop(columns=['temp', 'id', 'id_str'])
    else:
        df = df.drop(columns=['temp'])
    return df


def concat_label(data_type, source_feature_df):
    """Concat labels on source tweets
    data_type: 'dev', 'train'
    """
    df = pd.DataFrame(columns=['tweet_id', 'label'])
    with open(f'./data/original_data/{data_type}_source.txt', 'r') as f:
        ids = f.readlines()
    with open(f'./data/original_data/{data_type}.label.txt', 'r') as f:
        labels = f.readlines()
    df['tweet_id'] = [id.strip() for id in ids]
    df['label'] = [label.strip() for label in labels]
    df_labels = pd.merge(source_feature_df, df, on='tweet_id', how='left')
    df_labels['label'] = df_labels['label'].apply(lambda x: 0 if x == 'nonrumour' else 1)
    return df_labels

# dev_source_df = concat_label('dev', dev_source_df)
# train_source_df = concat_label('train', train_source_df)


def concat_reply(data_type, source_df):
    """concat replies on source tweets
    data_type: 'dev', 'train', 'test'
    """
    df = pd.DataFrame(columns=['tweet_id', 'reply'])
    with open(f'./data/original_data/{data_type}.data_sorted.txt', 'r') as f:
        content = f.readlines()
    df['tweet_id'] = [c.split(',')[0].strip() for c in content]
    df['reply'] = [','.join([i.strip() for i in c.split(',')[1:]]) for c in content]
    source_df = pd.merge(source_df, df, on='tweet_id', how='left')
    return source_df

# dev_source_df = concat_reply('dev', dev_source_df)
# train_source_df = concat_reply('train', train_source_df)

def check_weekday(df):
    """
    df: source_df or reply_df
    """
    df['isoweekday'] = df['created_at'].apply(lambda x: datetime.strptime(x.split('T')[0], '%Y-%m-%d').isoweekday())
    df['isweekday'] = df['isoweekday'].apply(lambda x: 1 <= x <= 5)
    df = df.drop(columns='isoweekday')
    return df
def check_weekday_test(df):
    """
    df: source_df or reply_df
    """
    df['isoweekday'] = df['created_at'].apply(lambda x: datetime.strptime(str(x).split(' ')[0], '%Y-%m-%d').isoweekday())
    df['isweekday'] = df['isoweekday'].apply(lambda x: 1 <= x <= 5)
    df = df.drop(columns='isoweekday')
    return df
def concat_user_info(data_type, source_df, user_json=None):
    """concat user info on source tweets
    data_type: 'dev', 'train'
    """
    if data_type != 'covid':
        df = json2df(f'./data/full data/{data_type}_source_userinfo.json')
    else:
        df = json2df(None, user_json)
    df['user_id'] = df.index
    source_df = pd.merge(source_df, df, on='user_id', how='left')
    return source_df

def concat_reply_info(reply_ids, reply_df, statis_feature):
    statistic_dict = dict()   
    replies_txt = ''
    reply_df = reply_df.fillna(0) # add
    if reply_ids == '':
        return [np.nan] * (len(statis_feature) + 1)
    for r in reply_ids.split(','):
        # reply_num = len(reply_ids.split(','))
        cur_reply_txt = reply_df.loc[r]['text'].strip()
        if cur_reply_txt != '':
            replies_txt += cur_reply_txt + ' [SEP] ' ## TODO
        ## get statistic features
        for f in statis_feature:
            statistic_dict[f] = statistic_dict.get(f, 0) + int(reply_df.loc[r][f])
    res_values = [replies_txt.strip()]
    for f in statis_feature:
        res_values.append(statistic_dict[f])
    return res_values

def processing_train_dev(data_type, isTrain=True):
    source_df, reply_df = clean_data(data_type)
    # source_df: 
    # ['text', 'reply_count', 'like_count', 'retweet_count', 'quote_count',
    # 'possibly_sensitive', 'created_at', 'user_id', 'has_url',
    # 'mentioned_url_num', 'id_num', 'tweet_id']

    # reply_df:
    # ['text', 'reply_count', 'like_count', 'retweet_count', 'quote_count',
    # 'possibly_sensitive', 'created_at', 'user_id', 'has_url',
    # 'mentioned_url_num', 'id_num', 'tweet_id']
    if isTrain:
        source_df = concat_label(data_type, source_df)
    source_df = concat_reply(data_type, source_df)
    source_df = check_weekday(source_df)
    source_df = concat_user_info(data_type, source_df)
    reply_df = check_weekday(reply_df)
    # add sentiment score
    source_df['senti_score'] = source_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df['senti_score'] = reply_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df.index = reply_df['tweet_id']
    source_df.index = source_df['tweet_id']
    # concat replies info to source_df
    # quote_count
    statis_feature = ['reply_count', 'like_count', 'retweet_count', 'quote_count',
                        'possibly_sensitive', 'has_url', 'mentioned_url_num', 
                        'id_num', 'isweekday', 'senti_score']
    source_df[['reply_text'] + ['reply_' + s for s in statis_feature]] = source_df.apply(lambda x: concat_reply_info(x['reply'], reply_df, statis_feature), axis=1, result_type='expand')                
    if isTrain:
        source_df = source_df.loc[~source_df['reply_text'].isnull()]
    
    return source_df

def processing_test():
    source_df = pd.read_json(path_or_buf='./data/tweet-objects/test_source.json', lines=True)
    
    reply_df= pd.read_json(path_or_buf='./data/tweet-objects/test_reply.json', lines=True)
    source_df = clean_test_data(source_df)
    reply_df = clean_test_data(reply_df)
    for i in ['verified', 'followers_count', 'listed_count']:
        source_df[i] = source_df['user'].apply(lambda x: x[i])
    source_df['has_url'] = source_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    # get reply statistic info
    reply_df['has_url'] = reply_df['entities'].apply(lambda x: 0 if len(x['urls']) == 0 else 1)
    source_df = concat_reply('test', source_df)
    source_df['reply_count'] = source_df['reply'].apply(lambda x: len(x.split(',')))
    source_df.index = source_df['tweet_id']
    with open('data/original_data/test_source.txt', 'r') as f:
        c = f.readlines()
    source_df = source_df.loc[[i.strip() for i in c]]
    source_df = check_weekday_test(source_df)
    reply_df = check_weekday_test(reply_df)
    # add sentiment score
    source_df['senti_score'] = source_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df['senti_score'] = reply_df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity > 0 else 0)
    reply_df.index = reply_df['tweet_id']
    reply_df = reply_df.rename(columns={'retweet_count': 'retweet_count', 'favorite_count': 'like_count',
                                        'mentioned_url_num': 'mentioned_url_num', 'id_num': 'id_num', 'isweekday': 'isweekday'})
    # source_df.index = source_df['tweet_id']
    source_df = source_df.rename(columns={'retweet_count': 'retweet_count', 'favorite_count': 'like_count', 'followers_count': 'followers_count',
                                        'mentioned_url_num': 'mentioned_url_num', 'id_num': 'id_num', 'isweekday': 'isweekday', 
                                        'verified': 'verified', 'listed_count': 'tweet_count'})
    # concat replies info to source_df
    # reply_count, quote_count
    statis_feature = [ 'like_count', 'retweet_count', 
                        'possibly_sensitive', 'has_url', 'mentioned_url_num', 
                        'id_num', 'isweekday', 'senti_score']
    source_df[['reply_text'] + ['reply_' + s for s in statis_feature]] = source_df.apply(lambda x: concat_reply_info(x['reply'], reply_df, statis_feature), axis=1, result_type='expand')          
    # source_df[['reply_reply_count', 'reply_quote_count', 'quote_count']] = 0      
    return source_df

def extract_stat_tweet_feat(istrain, df):
    # extract statistic features
    # reply_reply_count， reply_quote_count，quote_count
    statistic_features = ['reply_like_count',
       'reply_retweet_count', 'reply_possibly_sensitive',
       'reply_has_url', 'reply_mentioned_url_num', 'reply_id_num',
       'reply_isweekday', 'reply_senti_score', 'reply_count', 'like_count', 'retweet_count',
       'possibly_sensitive', 'has_url',
       'mentioned_url_num', 'id_num', 'isweekday', 'followers_count', 'tweet_count', 'verified',
       'senti_score' ]
    stat_feat_df = df[statistic_features]
    stat_feat_df.index = df['tweet_id']
    if istrain:
        tweet_df = df[['tweet_id', 'text', 'reply_text', 'label']]
    else:
        tweet_df = df[['tweet_id', 'text', 'reply_text']]
    # tweet_df = df.drop(columns=statistic_features)
    tweet_df.index = df['tweet_id']
    # convert into float
    for col in ['tweet_count', 'followers_count', 'verified']:
        stat_feat_df[col] = stat_feat_df[col].apply(lambda x: float(x))
    # fill nan using corresponding mean
    stat_feat_df = stat_feat_df.fillna(stat_feat_df.mean())
    return stat_feat_df, tweet_df
if __name__ == '__main__': 
    
    print('process train.=========')             
    train_df = processing_train_dev('train')
    print('process dev.=========')             
    dev_df = processing_train_dev('dev')
    print('process test.======')
    test_df = processing_train_dev('test', False)
    train_stat_feat_df, train_tweet_df = extract_stat_tweet_feat(True, train_df)
    dev_stat_feat_df, dev_tweet_df = extract_stat_tweet_feat(True, dev_df)
    test_stat_feat_df, test_tweet_df = extract_stat_tweet_feat(False, test_df)
    # print('process test.=========')             
    # test_df = processing_test()
    # test_stat_feat_df, test_tweet_df = extract_stat_tweet_feat(False, test_df)


    print('save data.=========')     
    dev_tweet_df.to_csv('./data/filtered data/dev_tweet_df.csv')
    dev_stat_feat_df.to_csv('./data/filtered data/dev_stat_feat_df.csv')
    train_tweet_df.to_csv('./data/filtered data/train_tweet_df.csv')
    train_stat_feat_df.to_csv('./data/filtered data/train_stat_feat_df.csv')
    test_tweet_df.to_csv('./data/filtered data/test_tweet_df.csv')
    test_stat_feat_df.to_csv('./data/filtered data/test_stat_feat_df.csv')
    