import os
import pandas as pd
import yaml
import argparse
from src.data.normalize_text import normalize
from src.data.dictionary import build_vocab


def load_data_from_dir(dir_path, normal_args=None, sent_dict=None, topic_dict=None):
    sent_path = dir_path + '/sentiments.txt'
    text_path = dir_path + '/sents.txt'
    topic_path = dir_path + '/topics.txt'

    i2sent = None
    i2topic = None

    if sent_dict is not None:
        i2sent = {i: sent for sent, i in sent_dict.items()}
    if topic_dict is not None:
        i2topic = {i: topic for topic, i in topic_dict.items()}

    if normal_args is None:
        normal_args = {}

    with open(text_path, 'r', encoding='utf-8') as pf:
        texts = pf.readlines()
        texts = [normalize(s, **normal_args) for s in texts]

    with open(sent_path, 'r', encoding='utf-8') as pf:
        sents = pf.readlines()
        sents = [int(i) for i in sents]
        if i2sent is not None:
            sents = [i2sent[i] for i in sents]

    with open(topic_path, 'r', encoding='utf-8') as pf:
        topics = pf.readlines()
        topics = [int(i) for i in topics]
        if i2topic is not None:
            topics = [i2topic[i] for i in topics]

    df = pd.DataFrame({"text": texts, "sentiment": sents, "topic": topics})

    return df


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-cp", default='configs/configs.yaml', help="path to configs")
    args = parser.parse_args()

    with open(args.config_path, 'r') as pf:
        configs = yaml.load(pf, Loader=yaml.FullLoader)

    normal_args = configs['DATA']['NORMALIZE']
    raw_path = configs['DATA']['RAW']['data_path']
    gen_cf = configs['DATA']['GENERAL']

    # sent_dict
    sent_dict = configs['DATA']['LABEL']['sentiment']
    topic_dict = configs['DATA']['LABEL']['topic']
    # path configure
    data_path = gen_cf['data_path']
    train_name = gen_cf['train_name']
    val_name = gen_cf['val_name']
    test_name = gen_cf['test_name']
    train_path = data_path + '/' + train_name
    val_path = data_path + '/' + val_name
    test_path = data_path + '/' + test_name

    # load raw data
    train_df = load_data_from_dir(raw_path + '/train', normal_args, sent_dict, topic_dict)
    val_df = load_data_from_dir(raw_path + '/dev', normal_args, sent_dict, topic_dict)
    test_df = load_data_from_dir(raw_path + '/test', normal_args, sent_dict, topic_dict)

    # save data
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # create vocabulary
    texts = train_df['text'].tolist()
    texts.extend(val_df['text'].tolist())
    seq_vocab, max_seq_len, max_char_len = build_vocab(texts, thresh_num=gen_cf['thresh_count_word'])
    seq_vocab.save(gen_cf['vocab_path'])

    print("Max sequence length: ", max_seq_len)
    print("Max char length: ", max_char_len)





