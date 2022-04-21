# split source reply ids
from utils import timer
@timer('ms')
def split_source_reply(txt_file):
  """
  txt_file: 'train.data.txt'
  """
  with open(txt_file) as f:
      ids = f.readlines()
  source_ids = []
  reply_ids = []
  source_txt_file = txt_file.split('.')[0] + '_source_data.txt'
  print(source_txt_file)
  reply_txt_file = txt_file.split('.')[0] + '_reply_data.txt'
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

from collections import defaultdict
import json
import pandas as pd

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



if __name__ == '__main__':
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