import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer


def read_data(train1_file_path, train2_file_path, data_cache_path, tokenizer, debug=False) -> str:
    train1_df = pd.read_csv(train1_file_path, header=None, sep='\t')
    train2_df = pd.read_csv(train2_file_path, header=None, sep='\t')
    train_df = pd.concat([train1_df, train2_df], axis=0)
    if debug:
        train_df = train_df.head(200)

    data_df = {'train': train_df}
    processed_data = {}

    for data_type, df in data_df.items():
        inputs = defaultdict(list)
        for i, row in tqdm(df.iterrows(), desc=f'Preprocessing {data_type} data', total=len(df)):
            label = row[2]
            sentence_a, sentence_b = row[0], row[1]
            build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer)

        processed_data[data_type] = inputs

    if not os.path.exists(os.path.dirname(data_cache_path)):
        os.makedirs(os.path.dirname(data_cache_path))
    with open(data_cache_path, 'wb') as f:
        pickle.dump(processed_data, f)

    return data_cache_path


def generate_data(data, corpus_file_path):
    with open(corpus_file_path, 'w', encoding='utf8') as f:
        for row in tqdm(data, total=len(data)):
            f.write(row + '\n')


def generate_vocab(total_data, vocab_file_path):
    total_tokens = [token for sent in total_data for token in sent.split()]
    counter = Counter(total_tokens)
    vocab = [token for token, freq in counter.items()]
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + vocab
    with open(vocab_file_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(vocab))


def reverse_data(df):
    data = df.apply(lambda x: x[0].strip() + ' ' + x[1].strip(), axis=1).values
    #reverse_data = df.apply(lambda x: x[1].strip() + ' ' + x[0].strip(), axis=1).values
    #all_data = np.concatenate([data, reverse_data], axis=0)
    return data


def build_bert_inputs(inputs, label, sentence_a, sentence_b, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])
    inputs['labels'].append(label)


def main():
    print('Starting concat data ...')
    train_a_path = './tcdata/gaiic_track3_round1_train_20210228.tsv'
    test_a_path = './tcdata/gaiic_track3_round1_testA_20210228.tsv'
    test_b_path = './tcdata/gaiic_track3_round1_testB_20210317.tsv'
    train_r2_path = './tcdata/gaiic_track3_round2_train_20210407.tsv'
    vocab_file_path = './user_data/vocab.txt'
    corpus_file_path = './user_data/duality_pair_pretrain_no_nsp.txt'
    data_cache_path = './user_data/data.pkl'
    train_a_df = pd.read_csv(train_a_path, sep='\t', header=None)
    test_a_df = pd.read_csv(test_a_path, sep='\t', header=None)
    test_b_df = pd.read_csv(test_b_path, sep='\t', header=None)
    train_r2_df = pd.read_csv(train_r2_path, sep='\t', header=None)

    train_a_data = reverse_data(train_a_df)
    test_a_data = reverse_data(test_a_df)
    test_b_data = reverse_data(test_b_df)
    train_r2_data = reverse_data(train_r2_df)

    total_data = np.concatenate([train_a_data, test_a_data, test_b_data, train_r2_data], axis=0)

    #assert len(total_data) == 900000, 'total examples should equal 900000'

    generate_vocab(total_data, vocab_file_path)
    generate_data(total_data, corpus_file_path)

    tokenizer = BertTokenizer.from_pretrained(vocab_file_path)
    read_data(train_a_path, train_r2_path, data_cache_path, tokenizer)


if __name__ == '__main__':
    main()
