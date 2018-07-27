# -*- coding:utf-8 -*-
# author : apollo2mars@gmail.com

from keras.models import Model  # need cudnn
from keras.layers import Input, LSTM, Embedding, Activation, Dense
from utils.squad_preprocess import context_question_text_preprocess
from keras.utils.np_utils import to_categorical
from solution_keras.layers.myLayer import MyLayer
from keras.layers.wrappers import Bidirectional
import keras.callbacks
from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD,Adam

"""
tuning params
"""
num_epoch = 400
batch_size = 512

"""
context average number of character is 123.22429519928848
context max number of character is 679
question average number of character is 10.249829358595155
question max number of character is 40
index of answer is index of word, not character
"""
max_context_length = 679  # 679 *1024
dim_lstm_context = 256
max_question_length = 40  # 40 * 64
dim_lstm_question = 64

dim_emb = 50

log_filepath = '/tmp/keras_squad_normal_log'


embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qst, pad_txt_dev_cnt, pad_txt_dev_qst, \
    idx_train_beg, idx_dev_beg = context_question_text_preprocess()

idx_train_beg_array = to_categorical(idx_train_beg, num_classes=max_context_length)
idx_dev_beg_array = to_categorical(idx_dev_beg, num_classes=max_context_length)


context_input = Input(shape=(max_context_length,), dtype='int32', name='context_input')
context_emb = Embedding(input_dim=vocab_size, output_dim=dim_emb, trainable=True)(context_input)
context_lstm = Bidirectional(LSTM(units=dim_lstm_context//2, return_sequences=True))(context_emb)
# todo : other text information

# question
question_input = Input(shape=(max_question_length,), dtype='int32', name='question_input')
question_emb = Embedding(input_dim=vocab_size, output_dim=dim_emb, trainable=True)(question_input)
question_lstm = Bidirectional(LSTM(units=dim_lstm_question//2))(question_emb)
# todo : average or attention

myLayer = MyLayer(output_dim=max_context_length, dim_lstm_context=dim_lstm_context, dim_lstm_question=dim_lstm_question)([context_lstm, question_lstm])

bn_myLayer = BatchNormalization()(myLayer)

# dense1 = Dense(1024, activation='relu')(bn_myLayer)
# dense2 = Dense(256, activation='relu')(dense1)
# all_output = Dense(1, activation='relu')(dense2)

dense1 = Dense(1024, activation='relu')(bn_myLayer)
all_output = Dense(max_context_length, activation='softmax')(dense1)

model = Model(inputs=[context_input, question_input], outputs=all_output)
model.summary()

"""
model compile
"""
opt = Adam(lr=0.1, clipnorm=1, clipvalue=0.5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy']) # https://keras.io/zh/metrics/
# model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy']) # https://keras.io/zh/metrics/
"""
model fit
"""
# tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
# # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的
# # 权值，每层输出值的分布直方图
# cbks = [tb_cb]
# history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=idx_train_beg_array, epochs=100, batch_size=256,
#             verbose=1, callbacks=cbks, validation_data=[[pad_txt_dev_cnt, pad_txt_dev_qst], idx_dev_beg_array])
history = model.fit(x=[pad_txt_train_cnt, pad_txt_train_qst], y=idx_train_beg_array, epochs=num_epoch, batch_size=batch_size,
            verbose=1, validation_data=[[pad_txt_dev_cnt, pad_txt_dev_qst], idx_dev_beg_array])
model.save('squad_beg_index.h5', overwrite=True)
# """
# model evaluate
# """
score = model.evaluate(x=[pad_txt_dev_cnt, pad_txt_dev_qst], y=idx_dev_beg_array, verbose=1)
print('Test score:', score[0])  # loss
print('Test accuracy;', score[1])  # accuracy