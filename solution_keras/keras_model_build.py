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

from keras.models import Model  # need cudnn
from keras.layers import Input, LSTM, Embedding, Activation, Dense, Dropout
from utils.squad_preprocess import context_question_text_preprocess
from keras.utils.np_utils import to_categorical
from solution_keras.layers.myLayer import MyLayer
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import SGD,Adam

"""
tuning params
"""
cnt_max_len = 300
qn_max_len = 30
num_epoch = 200
batch_size = 256
dim_lstm_context = 128
dim_lstm_question = 32
dim_emb = 50

dim_fc_beg = 301
dim_fc_end = 301
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
    idx_train_beg, idx_train_end, idx_dev_beg, idx_dev_end = context_question_text_preprocess(cnt_max_len, qn_max_len)

idx_train_beg_array = to_categorical(idx_train_beg, num_classes=cnt_max_len)
idx_dev_beg_array = to_categorical(idx_dev_beg, num_classes=cnt_max_len)
idx_train_end_array = to_categorical(idx_train_end, num_classes=cnt_max_len)
idx_dev_end_array = to_categorical(idx_dev_end, num_classes=cnt_max_len)

"""
context represent
"""
context_input = Input(shape=(cnt_max_len,), dtype='int32', name='context_input')
context_emb = Embedding(input_dim=vocab_size, output_dim=dim_emb, trainable=True)(context_input)
context_lstm = Bidirectional(LSTM(units=dim_lstm_context//2, return_sequences=True))(context_emb)
# todo : other text information

"""
query represent
"""
question_input = Input(shape=(qn_max_len,), dtype='int32', name='question_input')
question_emb = Embedding(input_dim=vocab_size, output_dim=dim_emb, trainable=True)(question_input)
question_lstm = Bidirectional(LSTM(units=dim_lstm_question//2))(question_emb)
# todo : attention

"""
self define layer: predict index
"""
inx_beg = MyLayer(output_dim=cnt_max_len, dim_lstm_context=dim_lstm_context, dim_lstm_question=dim_lstm_question)([context_lstm, question_lstm])
inx_end = MyLayer(output_dim=cnt_max_len, dim_lstm_context=dim_lstm_context, dim_lstm_question=dim_lstm_question)([context_lstm, question_lstm])

"""
batch normalization
"""
# bn_beg = BatchNormalization()(inx_beg)
# bn_end = BatchNormalization()(inx_end)
"""
fully connected
"""
dense_beg = Dense(dim_fc_beg, activation='relu', name='dense_begin')(inx_beg)
# dense_beg = Dense(dim_fc_beg, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='dense_begin')(bn_beg)
dropout_beg = Dropout(drop_beg)(dense_beg) # 舍弃率
idx_beg_output = Dense(cnt_max_len, activation='softmax')(dropout_beg)

dense_end = Dense(dim_fc_end, activation='relu', name='dense_end')(inx_end)
# dense_end = Dense(dim_fc_end, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='dense_end')(bn_end)
dropout_end = Dropout(drop_end)(dense_end)
idx_end_output = Dense(cnt_max_len, activation='softmax')(dropout_end)

model = Model(inputs=[context_input, question_input], outputs=[idx_beg_output, idx_end_output])
model.summary()

"""
model compile
"""
# lr : learning rate
# clipnorm
# clipvalue 0.5 : 所有的梯度会被变换到 -0.5 到 0.5 的空间
# opt = Adam(lr=0.1, clipnorm=1, clipvalue=0.5)
opt = Adam(lr=0.001)

# http://keras-cn.readthedocs.io/en/latest/other/metrics/
# 单个用以代表输出各个数据点上均值的值
# metrics 返回单个batch 上 该指标的结果, 不用于训练
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy']) # https://keras.io/zh/metrics/
# model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy']) # https://keras.io/zh/metrics/
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
#
# cbks = early_stopping

# validation_data : 选择超参数 which to evaluate the loss and any model metrics at the end of each epoch
history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=[idx_train_beg_array, idx_train_end_array], epochs=num_epoch, batch_size=batch_size,
            verbose=1, callbacks=[EarlyStopping(patience=10)], validation_split=0.1)
model.save('squad_all_index.h5', overwrite=True)
# """
# model evaluate
# """
score = model.evaluate(x=[pad_txt_dev_cnt, pad_txt_dev_qst], y=[idx_dev_beg_array, idx_dev_end_array], verbose=1)
print('Test score:', score[0])  # loss
print('Test accuracy;', score[1])  # accuracy