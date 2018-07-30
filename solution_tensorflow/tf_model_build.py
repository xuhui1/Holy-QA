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
tf.app.flags.DEFINE_bool("do_char_embedding", False, "Include char embedding - True/False")
tf.app.flags.DEFINE_bool("add_highway_layer", True, "Add highway layer to concatenated embeddings -Ture/False")
tf.app.flags.DEFINE_bool("cnn_encoder", False, "Add CNN Encoder Layer - True/False")
tf.app.flags.DEFINE_bool("rnet_attention", False, "Preform RNET QP and SM attentin - True/Flase")
tf.app.flags.DEFINE_bool("bidaf-attention", True, "Use BIDAF Attention-True/False")
tf.app.flags.DEFINE_bool("smart_span", True, "Select start and end idx based on smart conditions - True/False")

# Hyperparameters for Char CNN
tf.app.flags.DEFINE_integer("char_embedding_size", 8, "Size of char embedding")  # as suggested in handout
tf.app.flags.DEFINE_integer("word_max_len", 16, "max length for each word")  # 99th percentile from jupyter notebook
tf.app.flags.DEFINE_integer("char_out_size", 100, "num filters char CNN/out size")  # smae as filter size; as suggested in handout
tf.app.flags.DEFINE_integer("window_width", 5, "Kernel size for char cnn")  # as suggested in handout

# Hyperparametes for CNN Encoder
tf.app.flags.DEFINE_integer("filter_size_encoder", 50, "Size of cnn encoder")

# Hyperparameters for BIDAF
tf.app.flags.DEFINE_integer("hidden_size_modeling", 150, "Size of modeling layer")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per cauculating loss/f1/em on dev set. Warning:this is fairly time-consuming")

# Read and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Default to ./experiments/")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to ./data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint file.  You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializers model from train_dir.

    Inputs:
        session : TensorFlow session
        model : QAModel
        train_dir : path to directory where we'll look for checkpoint
        expect_exists : If True, throw an error if no checkpoint is found
                        If Flase, initialize fresh model if no checkpoint if found
    """
    print("Looking for model at {}".format(train_dir))
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at {}".format(train_dir))
        else:
            print("Thers is no saved checkpoint at {}".format(train_dir))
            session.run(tf.global_variables_initializer())
            print("Num params : {}".format(sum(v.get_shape().num_elements() for v in tf.trainable_variables())))

def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: {}".format(unused_argv))
    # check for Python 3
    if sys.version_info[0] != 3:
        raise Exception("Error: You must use python 3 but you are running Python {}".format(sys.version_info[0]))

    # Print out Tensorflow version
    print("This code war developed and tested an Tensorflow 1.4.1. Your Tensorflow version : {}".format(tf.__version__))

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove directory
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.context")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
    dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

    # Some GPU settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":
        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir), "log.txt")
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:
            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)
            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_context_path, dev_qn_path, dev_ans_path)

    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print("Writing predictions to %s..." % FLAGS.json_out_path)
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print("Wrote predictions to %s" % FLAGS.json_out_path)

    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()





