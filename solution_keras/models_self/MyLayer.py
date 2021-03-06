from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal


class MyLayer(Layer):
    def __init__(self, len_cnt, len_cnt_emb, len_qn_emb, **kwargs):
        self.len_cnt = len_cnt
        self.len_cnt_emb = len_cnt_emb
        self.len_qn_emb = len_qn_emb
        super(MyLayer, self).__init__(**kwargs)

    # 这是定义权重的方法，可训练的权应该在这里被加入列
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.len_cnt_emb, self.len_qn_emb),
                                      initializer=RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # 这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
    def call(self, x):

        all_cnt = x[0]  # context_padding_length, dim_lstm_context
        print("input context shape is {}".format(all_cnt.shape))
        all_qst = x[1]  # question_padding_length, dim_lstm_question
        print("input question shape is {}".format(all_qst.shape))
        tmp1 = K.dot(all_cnt, self.kernel)
        # tmp2 = tf.transpose(all_qst, perm=[1, 0])
        tmp3 = K.batch_dot(tmp1, all_qst)
        print("return tensor shape is {}".format(tmp3.shape))
        return tmp3
    # 如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
    def compute_output_shape(self, input_shape):
        # return (input_shape[0], self.len_cnt)
        print('input shape[0][0] is {}'.format(input_shape[0][0]))
        print('len_cnt is {}'.format(self.len_cnt))
        return (input_shape[0][0], self.len_cnt)

    def get_config(self):
        config = {'len_cnt': self.len_cnt,
                  'len_cnt_emb': self.len_cnt_emb,
                  'len_qn_emb': self.len_qn_emb}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))