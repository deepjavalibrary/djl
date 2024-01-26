class MyModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._02Linear_weight = tf.Variable(tf.random.normal(
        shape=[128, 784],
        mean=0.0,
        stddev=0.050507627,
        dtype=tf.dtypes.float32,
        name='normal_1_',
    ))
    self._02Linear_bias = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_2_',
    ))
    self._04Linear_weight = tf.Variable(tf.random.normal(
        shape=[64, 128],
        mean=0.0,
        stddev=0.125,
        dtype=tf.dtypes.float32,
        name='normal_3_',
    ))
    self._04Linear_bias = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_4_',
    ))
    self._06Linear_weight = tf.Variable(tf.random.normal(
        shape=[10, 64],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_5_',
    ))
    self._06Linear_bias = tf.Variable(tf.zeros(
        shape=[10],
        dtype=tf.dtypes.float32,
        name='zeros_6_',
    ))

## 4
  def call(self, x):
    result = tf.nn.bias_add(
        tf.matmul(
            tf.nn.relu(
                tf.nn.bias_add(
                    tf.matmul(
                        tf.nn.relu(
                            tf.nn.bias_add(
                                tf.matmul(
                                    tf.reshape(
                                        x, # (32, 1, 28, 28)
                                        shape=[-1, 784],
                                        name='reshape_7_',
                                    ), # (32, 784)
                                    b=self._02Linear_weight, # (128, 784)
                                    transpose_b=True,
                                    name='matmul_8_',
                                ), # (32, 128)
                                bias=self._02Linear_bias, # (128)
                                data_format=None,
                                name='bias_add_9_',
                            ), # (32, 128)
                            name='relu_10_',
                        ), # (32, 128)
                        b=self._04Linear_weight, # (64, 128)
                        transpose_b=True,
                        name='matmul_11_',
                    ), # (32, 64)
                    bias=self._04Linear_bias, # (64)
                    data_format=None,
                    name='bias_add_12_',
                ), # (32, 64)
                name='relu_13_',
            ), # (32, 64)
            b=self._06Linear_weight, # (10, 64)
            transpose_b=True,
            name='matmul_14_',
        ), # (32, 10)
        bias=self._06Linear_bias, # (10)
        data_format=None,
        name='bias_add_15_',
    )
    return result

## 4
def loss(label, prediction):
    result = tf.reduce_mean(
        tf.negative(
            tf.gather(
                tf.nn.log_softmax(
                    prediction, # (32, 10)
                    axis=-1,
                    name='log_softmax_16_',
                ), # (32, 10)
                indices=label, # (32)
                batch_dims=1,
                name='gather_17_',
            ), # (32, 1)
            name='negative_18_',
        ), # (32, 1)
        name='reduce_mean_19_',
    )
    return result

# number of epochs was 2
# number of batches was 32

