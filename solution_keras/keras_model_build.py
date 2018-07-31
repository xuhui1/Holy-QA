# -*- coding:utf-8 -*-
# author : apollo2mars@gmail.com
"""
SQuAD 1.1 train data information
context average number of character is 123.22429519928848
context max number of character is 679
question average number of character is 10.249829358595155
question max number of character is 40
index of answer is index of word, not character
"""

import os

from keras.models import Model, load_model  # need cudnn
from keras.layers import Input, LSTM, Embedding, Activation, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization

from utils.squad_preprocess import context_question_text_preprocess
from keras.utils.np_utils import to_categorical
from solution_keras.models_self.MyLayer import MyLayer
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras.optimizers import SGD,Adam

import numpy as np

"""
file settings
"""
model_name = "squad_all_index.h5"

"""
tuning params
"""
len_cnt_padding = 300
len_qn_padding = 30

num_epoch_single = 10
ALL_TESTS = 20
tmp_tests = 0

batch_size = 256
len_lstm_context = 128
len_lstm_question = 32
len_emb = 50

len_fc_beg = 301
len_fc_end = 301
drop_beg = 0.5
drop_end = 0.5

reg_beg_kernel = 1e-3
reg_end_kernel = 1e-3
reg_beg_activity = 1e-3
reg_end_activity = 1e-3

"""
tensorboard setting
"""
log_filepath = '/tmp/keras_squad_normal_log'

""""
data preprocess
"""
embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qst, pad_txt_dev_cnt, pad_txt_dev_qst, \
    idx_train_beg, idx_train_end, idx_dev_beg, idx_dev_end = context_question_text_preprocess(len_cnt_padding, len_qn_padding)

# idx_train_beg_array = to_categorical(idx_train_beg, num_classes=len_cnt_padding)
# idx_dev_beg_array = to_categorical(idx_dev_beg, num_classes=len_cnt_padding)
# idx_train_end_array = to_categorical(idx_train_end, num_classes=len_cnt_padding)
# idx_dev_end_array = to_categorical(idx_dev_end, num_classes=len_cnt_padding)

idx_train_beg = np.asarray(idx_train_beg)
idx_dev_beg = np.asarray(idx_dev_beg)
idx_train_end = np.asarray(idx_train_end)
idx_dev_end = np.asarray(idx_dev_end)


"""
context represent
"""
context_input = Input(shape=(len_cnt_padding,), dtype='int32', name='context_input')
context_emb = Embedding(input_dim=vocab_size, output_dim=len_emb, trainable=True)(context_input)
context_lstm = Bidirectional(LSTM(units=len_lstm_context//2, return_sequences=True))(context_emb)
# todo : other text information

"""
query represent
"""
question_input = Input(shape=(len_qn_padding,), dtype='int32', name='question_input')
question_emb = Embedding(input_dim=vocab_size, output_dim=len_emb, trainable=True)(question_input)
question_lstm = Bidirectional(LSTM(units=len_lstm_question//2))(question_emb)
# todo : attention

"""
self define layer: predict index
"""
inx_beg = MyLayer(len_cnt_padding, len_lstm_context, len_lstm_question)([context_lstm, question_lstm])
inx_end = MyLayer(len_cnt_padding, len_lstm_context, len_lstm_question)([context_lstm, question_lstm])

"""
batch normalization
"""
# bn_beg = BatchNormalization()(inx_beg)
# bn_end = BatchNormalization()(inx_end)
"""
fully connected
"""
dense_beg = Dense(len_fc_beg, activation='relu', name='dense_begin')(inx_beg)
# dense_beg = Dense(len_fc_beg, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='dense_begin')(bn_beg)
dropout_beg = Dropout(drop_beg)(dense_beg) # 舍弃率
idx_beg_output = Dense(len_cnt_padding, activation='softmax')(dropout_beg)

dense_end = Dense(len_fc_end, activation='relu', name='dense_end')(inx_end)
# dense_end = Dense(len_fc_end, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='dense_end')(bn_end)
dropout_end = Dropout(drop_end)(dense_end)
idx_end_output = Dense(len_cnt_padding, activation='softmax')(dropout_end)

model = Model(inputs=[context_input, question_input], outputs=[idx_beg_output, idx_end_output])
model.summary()

"""
model compile
"""
# lr : learning rate
# clipnorm
# clipvalue 0.5 : 所有的梯度会被变换到 -0.5 到 0.5 的空间
# opt = Adam(lr=0.1, clipnorm=1, clipvalue=0.5)
opt = Adam(lr=0.0001)

# http://keras-cn.readthedocs.io/en/latest/other/metrics/
# 单个用以代表输出各个数据点上均值的值
# metrics 返回单个batch 上 该指标的结果, 不用于训练
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy']) # https://keras.io/zh/metrics/
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']) # https://keras.io/zh/metrics/
"""
model fit
"""
# tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
# # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的
# # 权值，每层输出值的分布直方图
# cbks = [tb_cb]
# history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=idx_train_beg_array, epochs=100, batch_size=256,
#             verbose=1, callbacks=cbks, validation_data=[[pad_txt_dev_cnt, pad_txt_dev_qst], idx_dev_beg_array])

# early_stopping = EarlyStopping(patience=3)

# validation_data : 选择超参数 which to evaluate the loss and any model metrics at the end of each epoch

#
# 自定义的层不可以保存，但是如果自定义层在代码中已经存在，则可以正确重构。例如在载入之前运行一遍自定义层，然后在load_model时提供声明该层是custom object.

while tmp_tests < ALL_TESTS:
    tmp_tests += 1
    if os.path.exists(model_name):

        print("\n### Next Epochs ####\n")
        print("File exist, fit on exist model")
        print("Tests Batch {}".format(tmp_tests))

        load_model(model_name, custom_objects={'MyLayer': MyLayer(len_cnt=len_cnt_padding, len_cnt_emb=len_lstm_context, len_qn_emb=len_lstm_question)})

        history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=[idx_train_beg, idx_train_end], epochs=num_epoch_single, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
        model.save(model_name, overwrite=True)
        # """
        # model evaluate
        # """
        score = model.evaluate(x=[pad_txt_dev_cnt, pad_txt_dev_qst], y=[idx_dev_beg, idx_dev_end], verbose=1)
        print('Test score:', score[0])  # loss
        print('Test accuracy;', score[1])  # accuracy
    else:
        print("File not exist, train begin")
        print("Tests Batch {}".format(tmp_tests))

        history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=[idx_train_beg, idx_train_end], epochs=num_epoch_single,
            batch_size=batch_size, verbose=1, validation_split=0.1)
        model.save(model_name, overwrite=True)
        # """
        # model evaluate
        # """
        score = model.evaluate(x=[pad_txt_dev_cnt, pad_txt_dev_qst], y=[idx_dev_beg, idx_dev_end], verbose=1)
        print('Test score:', score[0])  # loss
        print('Test accuracy;', score[1])  # accuracy