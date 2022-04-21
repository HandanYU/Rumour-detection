# split source reply ids
from utils import timer
from collections import defaultdict
import json
import pandas as pd

@timer('ms')
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
# split_source_reply('data/original_data/dev.data.txt')
# split_source_reply('data/original_data/train.data.txt')
# split_source_reply('data/original_data/test.data.txt')

# crawl train and dev data
# !twarc2 hydrate  dev_reply.txt > dev_reply_data.jsonl
# !twarc2 hydrate  dev_source.txt > dev_source_data.jsonl
# !twarc2 hydrate  train_reply.txt > train_reply_data.jsonl
# !twarc2 hydrate  train_source.txt > train_source_data.jsonl
# resource: https://scholarslab.github.io/learn-twarc/06-twarc-command-basics.html


# get train and dev features
@timer('ms')
def filter_feature(jsonl_file_name):
    """
    jsonl_file_name: 'dev_source_data.jsonl'
    """
    json_file_name = '_'.join(jsonl_file_name.split('_')[:-1]) + '.json'
    json_data = pd.read_json(path_or_buf=jsonl_file_name, lines=True)
    data_dict = defaultdict(dict)
    for i in range(json_data.shape[0]):
        for j in range(len(json_data.data.iloc[i])):
            data_dict[json_data.data.iloc[i][j]['id']]['text'] = json_data.data.iloc[i][j]['text']
            data_dict[json_data.data.iloc[i][j]['id']]['reply_count'] = json_data.data.iloc[i][j]['public_metrics']['reply_count']
            data_dict[json_data.data.iloc[i][j]['id']]['like_count'] = json_data.data.iloc[i][j]['public_metrics']['like_count']
            data_dict[json_data.data.iloc[i][j]['id']]['retweet_count'] = json_data.data.iloc[i][j]['public_metrics']['retweet_count']
            data_dict[json_data.data.iloc[i][j]['id']]['quote_count'] = json_data.data.iloc[i][j]['public_metrics']['quote_count']
            data_dict[json_data.data.iloc[i][j]['id']]['possibly_sensitive'] = json_data.data.iloc[i][j]['possibly_sensitive']
            data_dict[json_data.data.iloc[i][j]['id']]['created_at'] = json_data.data.iloc[i][j]['created_at'] #add create time
            data_dict[json_data.data.iloc[i][j]['id']]['user_id'] = json_data.data.iloc[i][j]['author_id'] #add user id
            data_dict[json_data.data.iloc[i][j]['id']]['has_url'] = 1 if 'entities' in json_data.data.iloc[i][j] else 0 #add shared url number

    #  convert into json format
    dict_json=json.dumps(data_dict)
    # save json file
    with open(json_file_name, 'w+') as file:
        file.write(dict_json)
# filter_feature('dev_reply_data.jsonl')



@timer('ms')
def get_user_info(jsonl_file_name):
    """
    jsonl_file_name: 'dev_source_data.jsonl'
    """
    json_file_name = '_'.join(jsonl_file_name.split('_')[:-1]) + '_userinfo.json'
    json_data = pd.read_json(path_or_buf=jsonl_file_name, lines=True)
    # collect the user info
    info_dict = defaultdict(dict)
    for i in range(json_data.shape[0]):
        for j in range(len(json_data.includes.iloc[i]['users'])):
            info_dict[json_data.includes.iloc[i]['users'][j]['id']]['followers_count'] = json_data.includes.iloc[i]['users'][j]['public_metrics']['followers_count']
            info_dict[json_data.includes.iloc[i]['users'][j]['id']]['tweet_count'] = json_data.includes.iloc[i]['users'][j]['public_metrics']['tweet_count']
            info_dict[json_data.includes.iloc[i]['users'][j]['id']]['verified'] = json_data.includes.iloc[i]['users'][j]['verified']
    #  convert into json format
    dict_json=json.dumps(info_dict)
    # save json file
    with open(json_file_name, 'w+') as file:
        file.write(dict_json)


@timer('ms')
def sort_by_time(raw_file, json_file):
    with open(raw_file) as file:
        ids = file.readlines()

    with open(json_file, 'r+') as file:
        content = file.read()
        content=json.loads(content)
        df = pd.DataFrame(content)
        df = df.T

    save_name = raw_file[:-4] + '_sorted.txt'
    with open(save_name, 'w') as file:
        date = pd.Series(pd.DatetimeIndex(df.iloc[:, 6]), index=df.index)
        df.drop(['created_at'], axis=1, inplace=True)
        df['time'] = date

        for id_ in ids:
            ids_ = id_.strip().split(',')
            source_id = ids_[0]
            file.write(source_id)
            if len(ids_) > 1:
                reply_ids = ids_[1:]
                reply_ids[-1] = reply_ids[-1].replace('\n', '')
                valid_ids = [index for index in reply_ids if index in df.index]
                sorted_replies = df.loc[valid_ids].sort_values(by='time')
                if len(valid_ids) > 0:
                    file.write(',')

                for i, index in enumerate(sorted_replies.index):
                    file.write(index)
                    if i != len(sorted_replies.index) - 1:
                        file.write(',')

            file.write('\n')

def sort_by_time_test(raw_file, json_file):
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
if __name__ == '__main__':
    print('split reply and source:=========')
    split_source_reply('data/original_data/dev.data.txt')
    split_source_reply('data/original_data/train.data.txt')
    split_source_reply('data/original_data/test.data.txt')

    jsonls = ['./data/full data/dev_source_data.jsonl',
            './data/full data/dev_reply_data.jsonl',
            './data/full data/train_source_data.jsonl',
            './data/full data/train_reply_data.jsonl']
    print('filter feature:=====')
    for jsonl in jsonls:
        filter_feature(jsonl)
    jsonls = ['./data/full data/dev_source_data.jsonl',
            './data/full data/train_source_data.jsonl']
    print('get user info:=====')
    for jsonl in jsonls:
        get_user_info(jsonl)
    print('sort the replies')
    raw_files = ['data/original_data/train.data.txt', 'data/original_data/dev.data.txt']
    json_files = ['./data/full data/train_reply.json', './data/full data/dev_reply.json']
    for raw_file, json_file in zip(raw_files, json_files):
        sort_by_time(raw_file, json_file)
    raw_files = ['./data/original_data/test.data.txt']
    json_files = ['./data/tweet-objects/test_reply.json']
    for raw_file, json_file in zip(raw_files, json_files):
        sort_by_time_test(raw_file, json_file)