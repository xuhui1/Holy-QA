# -*- coding: utf-8 -*-
# reference : cs224n-squad
"""
This file contains the entropypoint to the rest of the code
"""

import os
import io
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers, generate_answers_prob

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname((os.path.abspath(__file__)))) # relative path of the main directory
print('main dir is {}'.format(MAIN_DIR))
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")  # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments")  #relative path of experiments dir

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple")
tf.app.flags.DEFINE_string("mode", "train", "Availabel modes: train/show_examples/official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for experiment. This will create a directory by this name in the experiments/ directory, which will hold related to this experiment")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradient to this norm")  # t_list[i] * clip_norm / max(global_norm, clip_norm)
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states")  # 150 for bidaf; 200 for others
tf.app.flags.DEFINE_integer("hidden_size_qp_matching", 150, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_sm_matching", 50, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 300, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100)

# Bool flags to select different models
tf.app.flags.DEFINE_bool("do_char_embedding", False, )



from utils.squad_preprocess import context_question_text_preprocess

embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qst, pad_txt_dev_cnt, pad_txt_dev_qst, \
    idx_train_beg, idx_dev_beg = context_question_text_preprocess()