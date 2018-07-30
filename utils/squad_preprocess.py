# -*- coding:utf-8 -*-
# apollo2mars@gamil.com
# do data preprocess for SQuAD
# input : SQuAD data
# output : some file after prepross

"""
SQuAD 数据集组成
536 篇文章, 23215个自然段, 107785个问题

file.json
├── "data"
│   └── [i]
│       ├── "paragraphs"
│       │   └── [j]
│       │       ├── "context": "paragraph text"
│       │       └── "qas"
│       │           └── [k]
│       │               ├── "answers"
│       │               │   └── [l]
│       │               │       ├── "answer_start": N
│       │               │       └── "text": "answer"
│       │               ├── "id": "<uuid>"
│       │               └── "question": "paragraph question?"
│       └── "title": "document id"
└── "version": 1.1
"""

import os
import argparse
import json
import nltk
import numpy as np
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as T

def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = ''  # accumulator eg : 'a', 'ap', 'app', 'appl', 'apple'
    current_token_idx = 0  # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context):  # step through original characters
        if char != u' ' and char != u'\n':  # if it's not a space:
            acc += char  # add to accumulator

            # context_token = unicode(context_tokens[current_token_idx])  # current word token
            context_token = context_tokens[current_token_idx]  # current word token, 读取词,

            if acc == context_token: # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                    # if context = "hello world" and context_tokens = ["hello", "world"] then
                    # 0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
                    # mapping[0] = ("hello", 0)
                    # mapping[1] = ("hello", 0)
                    # mapping[3] = ("hello", 0)
                    # mapping[4] = ("hello", 0)
                    # skip
                    # mapping[6] = ("world", 1)
                    # mapping[7] = ("world", 1)
                    # mapping[8] = ("world", 1)
                    # mapping[9] = ("world", 1)
                    # mapping[10] = ("world", 1)
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


# get context, question, answer pair
def preprocess_and_write(dataset, tier, out_dir):
    """
    reads the dataset
    extracts context, question, answer
    tokenizes them
    calculate answer span in terms of token indices
    
    Note: due to tokenization issues, and the fact that the original answer
    spans are given in terms of characters, some examples are discarded because
    we cannot get a clean span in terms of tokens.
    
    Inputs:
      dataset: read from JSON
      tier: string ("train" or "dev")
      out_dir: directory to write the preprocessed files
    Returns:
      the number of (context, question, answer) triples written to file by the dataset.
    """
    num_exs = 0 # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []
    
    for articles_id in range(len(dataset['data'])) :
        article_paragraphs = dataset['data'][articles_id]['paragraphs']  # 得到一系列段落
        for pid in range(len(article_paragraphs)):  # 遍历各个段落
            # context = unicode(article_paragraphs[pid]['context'])
            context = article_paragraphs[pid]['context']  # 获得单个段落的正文

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')  # 替换标点
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)  # list of strings (lowercase) 分词, 处理引号, 变小写
            context = context.lower()  # context 变小写
            
            qas = article_paragraphs[pid]['qas']  # list of questions 每个段落的一系列问题

            # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
            # 字符位置　映射到　词, 词下标
            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens) 

            if charloc2wordloc is None:  # there was a problem
                num_mappingprob += len(qas)  # 记录有多少个问题未被正常处理
                continue # skip this context example
                
            # for each question, process the question and answer and write to file
            for qn in qas:

                # read the question text and tokenize
                # question = unicode(qn['question']) # string
                question = qn['question'] # string 单独的一个问题
                question_tokens = tokenize(question) # list of strings 符号处理 分词 变小写

                # of the three answers, just take the first
                ans_text = qn['answers'][0]['text'].lower() # get the answer text 以第一个答案为正确答案
                # ans_text = unicode(qn['answers'][0]['text']).lower() # get the answer text
                ans_start_charloc = qn['answers'][0]['answer_start'] # answer start loc (character count) 在数据中已经给出
                ans_end_charloc = ans_start_charloc + len(ans_text) # answer end loc (character count) (exclusive)

                # Check that the provided character spans match the provided answer text 记录有多少问题的起始值是错误的
                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                  # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                  # We should upgrade to Python 3 next year!
                  num_spanalignprob += 1
                  continue

                # get word locs for answer start and end (inclusive)
                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] # answer start word loc 由字符下标得到词下标
                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1] # answer end word loc
                assert ans_start_wordloc <= ans_end_wordloc

                # Check retrieved answer tokens match the provided answer text.
                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                # and the answer character span is around "generation",
                # but the tokenizer regards "fifth-generation" as a single token.
                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue # skip this question/answer pair

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                num_exs += 1
    
    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))
    
    # shuffle examples
    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, tier +'.context'), 'w') as context_file,\
            open(os.path.join(out_dir, tier +'.question'), 'w') as question_file,\
            open(os.path.join(out_dir, tier +'.answer'), 'w') as ans_text_file,\
            open(os.path.join(out_dir, tier +'.span'), 'w') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default='./data/')
    return parser.parse_args()


def write_to_file(out_file, line):
    out_file.write(line.encode('utf8').decode() + '\n')


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]  # 分词, 对引号进行替换, 变为小写字母
    return tokens


def total_exs(dataset):
    """
    Returns the total number of (context, question, answer) triples,
    given the data read from the SQuAD json file.
    """
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def main():
    args = setup_args()

    print("Will download SQuAD datasets to {}".format(args.data_dir))
    print("Will put preprocessed SQuAD datasets in {}".format(args.data_dir))

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    train_filename = "../data/train-v1.1.json"
    dev_filename = "../data/dev-v1.1.json"

#   # download train set
#   maybe_download(SQUAD_BASE_URL, train_filename, args.data_dir, 30288272L)

    # read train set
    train_data = data_from_json(os.path.join(args.data_dir, train_filename))
    print("Train data has %i examples total" % total_exs(train_data))

    # preprocess train set and write to file
    preprocess_and_write(train_data, 'train', args.data_dir)

#   # download dev set
#   maybe_download(SQUAD_BASE_URL, dev_filename, args.data_dir, 4854279L)

    # read dev set
    dev_data = data_from_json(os.path.join(args.data_dir, dev_filename))
    print("Dev data has %i examples total" % total_exs(dev_data))

    # preprocess dev set and write to file
    preprocess_and_write(dev_data, 'dev', args.data_dir)


# load the whole embedding into memory
def load_emb():
    f = open('../data/glove.6B.50d.txt')
    embeddings_index = dict()
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


# read text from file
def read_txt_from_file(file_path):
    tmp_list = []
    with open(file_path) as f:
        for tmp_line in f:
            tmp_list.append(tmp_line)
    return tmp_list


# read index from file
def read_index_from_file(file_path, cnt_max_len):
    label_list_beg = []
    label_list_end = []
    with open(file_path) as f:
        for tmp_line in f:
            tmp_beg = int(tmp_line.split(" ")[0])
            tmp_end = int(tmp_line.split(" ")[1])

            if tmp_end < cnt_max_len and tmp_beg < cnt_max_len:
                label_list_beg.append(tmp_beg)
                label_list_end.append(tmp_end)
            else:  # todo : about 1% index is 0, and need to find some methods to solve this problem
                label_list_beg.append(0)
                label_list_end.append(0)
    return label_list_beg, label_list_end


def context_question_text_preprocess(cnt_max_len, qn_max_len):
    """
    get corpus
    """
    # file path name
    file_train_context = '../data/train.context'
    file_train_question = '../data/train.question'
    file_dev_context = '../data/dev.context'
    file_dev_question = '../data/dev.question'
    file_train_span = '../data/train.span'
    file_dev_span = '../data/dev.span'

    # text and index list
    txt_train_cnt = read_txt_from_file(file_train_context)
    txt_train_qst = read_txt_from_file(file_train_question)
    txt_dev_cnt = read_txt_from_file(file_dev_context)
    txt_dev_qst = read_txt_from_file(file_dev_question)
    idx_train_beg, idx_train_end = read_index_from_file(file_train_span, cnt_max_len)
    idx_dev_beg, idx_dev_end = read_index_from_file(file_dev_span, cnt_max_len)

    cnt_all_txt = txt_train_cnt+txt_dev_cnt
    qst_all_txt = txt_train_qst+txt_dev_qst

    # from keras.preprocessing.text import Tokenizer
    # 求 context 和 question 的长度列表
    l_cnt = list(map(lambda x: len(T.text_to_word_sequence(x)), cnt_all_txt))
    l_qst = list(map(lambda x: len(T.text_to_word_sequence(x)), qst_all_txt))

    # 求 context 和 question 的平均长度(词)
    import functools
    l_all_cnt = functools.reduce(lambda x, y: x+y, l_cnt)
    l_all_qst = functools.reduce(lambda x, y: x+y, l_qst)
    l_average_cnt = l_all_cnt/len(cnt_all_txt)
    l_average_qst = l_all_qst/len(qst_all_txt)

    # 分词
    t = Tokenizer()  # 分词器
    txt_list = txt_train_cnt + txt_train_qst + txt_dev_cnt + txt_dev_qst
    t.fit_on_texts(txt_list)
    vocab_size = len(t.word_index) + 1

    enc_txt_train_cnt = t.texts_to_sequences(txt_train_cnt)
    enc_txt_train_qst = t.texts_to_sequences(txt_train_qst)
    enc_txt_dev_cnt = t.texts_to_sequences(txt_dev_cnt)
    enc_txt_dev_qst = t.texts_to_sequences(txt_dev_qst)

    pad_txt_train_cnt = pad_sequences(enc_txt_train_cnt, maxlen=cnt_max_len, padding='post')
    pad_txt_train_qst = pad_sequences(enc_txt_train_qst, maxlen=qn_max_len, padding='post')
    pad_txt_dev_cnt = pad_sequences(enc_txt_dev_cnt, maxlen=cnt_max_len, padding='post')
    pad_txt_dev_qst = pad_sequences(enc_txt_dev_qst, maxlen=qn_max_len, padding='post')

    # load embedding
    embeddings_index = load_emb()

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 50))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("context average number of character is {}".format(l_average_cnt))
    print("context max number of character is {}".format(cnt_max_len))
    print("question average number of character is {}".format(l_average_qst))
    print("question max number of character is {}".format(qn_max_len))

    print("index of answer is index of word, not character")

    return embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qst, pad_txt_dev_cnt, pad_txt_dev_qst, \
           idx_train_beg, idx_train_end, idx_dev_beg, idx_dev_end


if __name__ == '__main__':
    main()
    # context_question_text_preprocess()
