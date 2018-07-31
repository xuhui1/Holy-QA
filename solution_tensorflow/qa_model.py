
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

        ################################################## CHAR EMBEDDING ##################################################
        if self.FLAGS.do_char_embed:
            self.context_emb_out, self.qn_emb_out = self.add_char_embeddins()

            self.context_embs = tf.concat((self.context_embs, self.context_emb_out), axis=2)
            print("Shape - concatenated context embs", self.context_embs.shape)

            self.qn_embs = tf.concat((self.qn_embs,self.qn_emb_out), axis=2)
            print("Shape - concatenated qn embs", self.qn_embs.shape)

        ################################################## HIGHWAY LAYER ##################################################
        if self.FLAGS.add_highway_layer:
            last_dim_concat = self.context_embs.get_shape().as_list()[-1]
            for i in range(2):
                # add two highway layers or repeat process twice
                self.context_embs = self.highway(self.context_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)
                # reuse variables for qu embs
                self.qn_embs = self.highway(self.qn_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)

        ################################################## RNN ENCODER ##################################################
        encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scopename='RNNEncoder') # shape (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scopename='RNNEncoder') # shape (batch_size, context_len, hidden_size*2)

        ################################################## CNN ENCODER ##################################################
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

        ################################################## RNET QUESTION CONTEXT ATTENTION and SELF ATTENTION ##################################################
        if self.FLAGS.rnet_attention: ## Perform Question Passage and Self Matching attention from R-Net
            rnet_layer = Attention_Match_RNN(self.keep_prob, self.FLAGS.hidden_size_encoder, self.FLAGS.hidden_size_qp_matching, self.FLAGS.hidden_size_sm_matching)
            # Implement better question_passage matching
            v_P = rnet_layer.build_graph_qp_matching(context_hiddens, question_hiddens, self.qn_mask, self.context_mask, self.FLAGS.context_len, self.FLAGS.question_len)

            self.rnet_attention = v_P
            self.rnet_attention = tf.sqeeze(self.rnet_attention, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            _, self.rnet_attention_pros = masked_softmax(self.rnet_attention, self.context_mask, 1)

            h_P = rnet_layer.build_graph_sm_matching(context_hiddens, question_hiddens, self.qn_mask, self.context_mask, self.FLAGS.context_len, self.FLAGS.question_len, v_P)

            # Blended reps for R-Net
            blended_reps = tf.concat([context_hiddens, v_P, h_P])

        ################################################## BIDAF ATTENTION AND MODELING LAYER ##################################################
        elif self.FLAGS.bidaf_attention:

            attn_layer = BiDAF(self.keep_prob, self.FLAGS.hidden_size_encoder*2)
            # attn_output is shape (batch_size, context_len, hidden_size_encoder*6)
            attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask)

            self.bidaf_attention = attn_output
            self.bidaf_attention = tf.reduce_max(self.bidaf_attention, axis=2) # shape (batch_size, seq_len)
            print("Shape bidaf before softmax", self.bidaf_attention.shape)

            # Take softmax over sequence
            _, self.bidaf_attention_probs = masked_softmax(self.bidaf_attention, self.context_mask, 1) # for plotting purpose

            blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size_encoder*8)

            ## add a modeling layer
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            attention_hidden = modeling_layer.build_graph(blended_reps,
                                                          self.context_mask, scopename='bidaf_modeling') # (batch_size, context_len, hidden_size*2)

            blended_reps = attention_hidden # for the final layer

        ################################################## BASELINE DOT PRODUCT ATTENTION ##################################################
        else: ## perform baseline dot product attention
            # Use context hidden states to attend to question hidden states - Basic Attention
            last_dim = context_hiddens.get_shape().as_list()[-1]
            print("last dim fo context hiddens {}".format(last_dim))

            attn_layer = BasicAttn(self.keep_prob, last_dim, last_dim)
            _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens)
            # attn_output si shape (batch_size, context_len, hidden_size*2)

            # Concat attn_output to context_hiddens to get blended_reps
            blended_reps = tf.concat([context_hiddens, attn_output]) # (batch_size, context_len, hidden_size*4)

        ################################################## ################################################## ##################################################
        #
        #                                                   Answering Network
        #
        ################################################## ################################################## ##################################################

        ################################################## RNET QUESTION POOLING and ANSWER POINTER ##################################################
        if self.FLAGS.answer_pointer_RNET: ## Use Answer Pointer Module from R-NET
            if self.FLAGS.rnet_attention:
                # different attention size for R-NET final layer
                # combined size of blended reps
                hidden_size_attn = 2*self.FLAGS.hidden_size_encoder + self.FLAGS.hidden_size_qp_matching + 2 * self.FLAGS.hidden_size_sm_matching

            elif self.FLAGS.bidaf_attention:
                hidden_size_attn = 2*self.FLAGS.hidden_size_modeling
            else :
                hidden_size_attn = 4 * self.FLAGS.hidden_size_encoder # Final attention size for baseline model

            attn_ptr_layer = Answer_Pointer(self.keep_prob, self.FLAGS.hidden_size_encoder,
                                            self.FLAGS.question_len, hidden_size_attn)
            p, logits = attn_ptr_layer.build_graph_answer_pointer(context_hiddens, question_hiddens, self.qn_mask, self.context_mask,
                                                                  self.FLAGS.context_len, self.FLAGS.question_len, blended_reps)

            self.logits_start = logits[0]
            self.probdist_start = p[0]
            self.logits_end = logits[0]
            self.probdist_end = p[1]

        ################################################## BASELINE FULLY CONNECTED ANSWER PREDICTION ##################################################

        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default

        # blended_reps_final is shape (batch_size, context_len, hidden_size)
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size_fully_connected)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(
                blended_reps_final,
                self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(
                blended_reps_final,
                self.context_mask)

    def add_loss(self):
        """
        Add loss computation to the graph
        Uses:
            self.logits_start: shape(batch_size, context_len)
              IMPORTANT : Assume that self.logits start is maksed (i.e. has -large in masked locations)
              That's because the tf.nn.sparse_softmax_cross_entorpy_with_logits function applies softmax and then computes cross-entropy loss.
              So you need to apply masking to the logits (by sbutracting large number in the padding location) before you pass to the sparse_softmax_cross_entropy_with_logits function.

            self.ans_span: shape(batch_size, 2)
              Contains the gold start and end locations

            Defines:
              self.loss_start, self.loss.end, sefl.loss : all scalar tensors
        """
        with vs.variable_scope("loss"):
            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:,0])# loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar("loss_start", self.loss_start) # log to tensorboard