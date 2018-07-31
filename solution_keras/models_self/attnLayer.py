from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal


class AttnLayer(Layer):
    def __init__(self, output_dim, dim_lstm_context, dim_lstm_question, **kwargs):
        self.output_dim = output_dim
        self.dim_lstm_context = dim_lstm_context
        self.dim_lstm_question = dim_lstm_question
        super(MyLayer, self).__init__(**kwargs)

    # 这是定义权重的方法，可训练的权应该在这里被加入列
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.ws = self.add_weight(name='similarity',
                                      shape=(self.dim_lstm_context * 6),
                                      initializer=RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      trainable=True)

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.dim_lstm_context * 2),
                                      initializer=RandomNormal(mean=0.0, stddev=0.5, seed=None),
                                      trainable=True)

        super(AttnLayer, self).build(input_shape)  # Be sure to call this somewhere!

    # 这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
    def call(self, x):

        batch_cnt = x[0]  # context H
        print("input context shape is {}".format(batch_cnt.shape))
        batch_qst = x[1]  # query U
        print("input question shape is {}".format(batch_qst.shape))

        ## Compute Stj

        ## Context 2 Query Attention
        # ai = softmax(St:)

        ## Query 2 Context Attention

        return tmp3
    # 如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
    def compute_output_shape(self, input_shape):
        # return (input_shape[0], self.output_dim)
        print('input shape[0][0] is {}'.format(input_shape[0][0]))
        print('output_dim is {}'.format(self.output_dim))
        return (input_shape[0][0], self.output_dim)