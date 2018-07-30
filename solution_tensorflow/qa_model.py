
# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import  *

import sys

from imp import reload
reload(sys)
sys.setdefaultencoding('utf-8')

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Topic-level Question Answering module"""
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializers the QA model.
        :param FLAGS:  the FLAGs passed from tf_model_build.py
        :param id2word: dictionary mapping word idx (int) to word(string)
        :param word2id: dictionary mapping word (string) to word idx(int)
        :param emb_matrix: numpy array shape (400002, embedding_size) containing pre-trained Glove embedding
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        _, _, num_chars = self.create_char_dicts()
        self.char_vacab = num_chars

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):

    def add_placeholder(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs
        # These are all batch-first:the None corresponds to batch_size and allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout)
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_drop = tf.placeholder_with_default(1.0, shape=())

        # For Char CNN
        self.char_ids_context = tf.placeholder(tf.int32, shape=(None, self.FLAGS.context_len, self.FLAGS.word_max_len))
        self.char_ids_qn = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_max_len])

    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to graph
        :param emb_matrix:  emb_matrix : shape(400002, embedding_size).
                                The Glove vectors, plus vectors for PAD and UNK.
        """

        with vs.variable_scope("embeddings"):
            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
            # Get the word embeddings for the context and question, using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, context_len, embedding_size)

    def add_char_embeddings(self):
        """
        Adds char embedding layer to the graph
        """

        def conv1d(input_, output_size, width, stride, scope_name):
            """

            :param input_: A tensor of embedded tokens shape [batch_size, max_length, embedding_size]
            :param output_size: The number of feature maps we'd like to calculate
            :param width: The filter width
            :param stride: The stride
            :param scope_name:
            :return: A tensor of with shape [batch_size, max_length, output_size]
            """
            inputSize = input_.get_shape()[-1] # How many channels on the input(the size of our embedding for instance)
            # This is the kicker where we make our text an image of height 1
            input_ = tf.expand_dims(input_, axis=1) # Change the shape to [batch_size, 1, max_length, output_size]
            # Make sure the height of the filter is 1
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                filter_ = tf.get_variable("conv_filter", shape=[1, width, inputSize, output_size])
            # Run the convolution as if this were an image
            convolved = tf.nn.conv2d(input_, filter=filter_, strides=[1,1,stride,1], padding="VALID")
            # Remove the extra dimension, eg make the shape [batch_size, max_length, output_size]
            result = tf.squeeze(convolved, axis=1) # tf.squeeze : Removes dimensions of size 1 from the shape of a tensor
            return result

        with vs.variable_scope("char_embedding"):
            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            char_emb_matrix = tf.Variable(tf.random_uniform((self.char_vocab, self.FLAGS.char_embedding_size), -1, 1)) # is trainable
            print("Shape context placeholder", self.char_ids_context.shape)
            print("Shape qn placeholder", self.char_ids_qn.shape)

            # shape(-1, word_max_len, char_embedding_size)
            self.context_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, tf.reshape(self.char_ids_context, shape=(-1, self.FLAGS.word_max_len)))

            ## reshape to 3d tensor - compress dimensions we don't to convolve on
            # shape batch_size*context_len, word_max_len, char_embedding_size
            self.context_char_embs = tf.reshape(self.context_char_embs, shape=(-1, self.FLAGS.word_max_len, self.FLAGS.char_embedding_size))
            print("Shape context embs before conv", self.context_char_embs.shape)

            ## Repeat for question embeddings - again reshape to 3D tensor
            self.qn_char_embs = embedding_ops.embedding_lookup(char_emb_matrix, tf.reshape(self.char_ids_qn, shape=(-1, self.FLAGS.word_max_len)))
            self.qn_char_embs = tf.reshape(self.qn_char_embs, shape=(-1, self.FLAGS.word_max_len, self.FLAGS.char_embedding_size))
            print("Shape qn embs before conv", self.qn_char_embs.shape)

            ## Now implement convolution. I decided to use conv2d through the function conv1d above since that was more intuitive
            self.context_emb_out = conv1d(input_= self.context_char_embs, output_size = self.FLAGS.char_out_size, width= self.FLAGS.window_width, stride=1, scope_name="char-rnn")
            self.context_emb_out = tf.nn.dropout(self.context_emb_out, self.keep_drop)
            print("Shape context embs after conv", self.context_emb_out.shape)

            self.context_emb_out = tf.reduce_sum(self.context_emb_out, axis=1)
            # Desired shape is Batch_size, context_len, char_out_size
            self.context_emb_out = tf.reshape(self.context_emb_out, shape=(-1, self.FLAGS.context_len, self.FLAGS.char_out_size))
            print("Shape context embs after pooling", self.context_emb_out.shape)


            self.qn_emb_out = conv1d(input_=self.qn_char_embs, output_size=self.FLAGS.char_out_size,
                                          width=self.FLAGS.window_width, stride=1, scope_name='char-cnn') #reuse weights b/w context and ques conv layers
            self.qn_emb_out = tf.nn.dropout(self.qn_emb_out, self.keep_prob)
            print("Shape qn embs after conv", self.qn_emb_out.shape)
            self.qn_emb_out = tf.reduce_sum(self.qn_emb_out,axis=1)
            self.qn_emb_out = tf.reshape(self.qn_emb_out, shape=(-1, self.FLAGS.question_len,
                                                                           self.FLAGS.char_out_size))  # Desired shape is Batch_size, question_len, char_out_size
            print("Shape qn embs after pooling", self.qn_emb_out.shape)


            return self.context_emb_out, self.qn_emb_out

def build_graph(self):
    """
    Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span
    Define:
        self.logits_starts, self.logits_end:Both tensors shape (batch_size, context_len)
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are - large in the pad locations. Necessary for when we feed into the entropy function.
        self.probdist start, self.probdist_end: Both shape (batch_size, context_len), Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
    """

    # Use a RNN to get hidden states for the context and question
    # Note : here the RNNEncoder is shared(i.e. the weights are the same) between the context and the question

    ##### CHAR EMBEDDING #####
    if self.FLAGS.do_char_embed:
        self.context_emb_out, self.qn_emb_out = self.add_char_embeddins()

        self.context_embs = tf.concat((self.context_embs, self.context_emb_out), axis=2)
        print("Shape - concatenated context embs", self.context_embs.shape)

        self.qn_embs = tf.concat((self.qn_embs,self.qn_emb_out), axis=2)
        print("Shape - concatenated qn embs", self.qn_embs.shape)

    ##### HIGHWAY LAYER #####
    if self.FLAGS.add_highway_layer:
        last_dim_concat = self.context_embs.get_shape().as_list()[-1]
        for i in range(2):
            # add two highway layers or repeat process twice
            self.context_embs = self.highway(self.context_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)
            # reuse variables for qu embs
            self.qn_embs = self.highway(self.qn_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)

    ##### RNN ENCODER #####
    encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
    context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scopename='RNNEncoder') # shape (batch_size, context_len, hidden_size*2)
    question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scopename='RNNEncoder') # shape (batch_size, context_len, hidden_size*2)

    ##### CNN ENCODER #####
    if self.FLAGS.cnn_encoder:
        ## Use CNN to also generate encodings
        cnn_encoder = CNNEncoder(self.FLAGS.filter_size_encoder, self.keep_prob)
        context_cnn_hiddens = cnn_encoder.build_graph(self.context_embs, self.FLAGS.context_len, scope_name='context-encoder')
        print("Shape - Context Encoder output", context_cnn_hiddens.shape)

        ques_cnn_hiddens = cnn_encoder.build_graph(self.qn_embs, self.FLAGS.question_len,scope_name='ques-encoder')  # (batch_size, context_len, hidden_size*2)
        print("Shape - Ques Encoder output", ques_cnn_hiddens.shape)

        ## concat these vectors
        context_hiddens = context_cnn_hiddens
        question_hiddens = ques_cnn_hiddens
        print("Shape - Context Hiddens", context_hiddens.shape)

    ##### RNET QUESTION CONTEXT ATTENTION and SELF ATTENTION #####
