class MyModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 3, 3, 3],
                mean=0.0,
                stddev=0.27216554,
                dtype=tf.dtypes.float32,
                name='normal_1_',
            ), # (64, 3, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_2_',
        ) # (3, 3, 3, 64)
    )
    self._02ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 64, 1, 1],
                mean=0.0,
                stddev=0.17677669,
                dtype=tf.dtypes.float32,
                name='normal_3_',
            ), # (64, 64, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_4_',
        ) # (1, 1, 64, 64)
    )
    self._02ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_5_',
        ) # (64)
    )
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_6_',
        ) # (64)
    )
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_7_',
        ) # (64)
    )
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_8_',
        ) # (64)
        , trainable = False
    )
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_9_',
        ) # (64)
        , trainable = False
    )
    self._02ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 64, 3, 3],
                mean=0.0,
                stddev=0.058925565,
                dtype=tf.dtypes.float32,
                name='normal_10_',
            ), # (64, 64, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_11_',
        ) # (3, 3, 64, 64)
    )
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_12_',
        ) # (64)
    )
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_13_',
        ) # (64)
    )
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_14_',
        ) # (64)
        , trainable = False
    )
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_15_',
        ) # (64)
        , trainable = False
    )
    self._02ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 64, 1, 1],
                mean=0.0,
                stddev=0.17677669,
                dtype=tf.dtypes.float32,
                name='normal_16_',
            ), # (256, 64, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_17_',
        ) # (1, 1, 64, 256)
    )
    self._02ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_18_',
        ) # (256)
    )
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_19_',
        ) # (256)
    )
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_20_',
        ) # (256)
    )
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_21_',
        ) # (256)
        , trainable = False
    )
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_22_',
        ) # (256)
        , trainable = False
    )
    self._02ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 64, 1, 1],
                mean=0.0,
                stddev=0.17677669,
                dtype=tf.dtypes.float32,
                name='normal_23_',
            ), # (256, 64, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_24_',
        ) # (1, 1, 64, 256)
    )
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_25_',
        ) # (256)
    )
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_26_',
        ) # (256)
    )
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_27_',
        ) # (256)
        , trainable = False
    )
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_28_',
        ) # (256)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_29_',
            ), # (64, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_30_',
        ) # (1, 1, 256, 64)
    )
    self._03ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_31_',
        ) # (64)
    )
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_32_',
        ) # (64)
    )
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_33_',
        ) # (64)
    )
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_34_',
        ) # (64)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_35_',
        ) # (64)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 64, 3, 3],
                mean=0.0,
                stddev=0.058925565,
                dtype=tf.dtypes.float32,
                name='normal_36_',
            ), # (64, 64, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_37_',
        ) # (3, 3, 64, 64)
    )
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_38_',
        ) # (64)
    )
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_39_',
        ) # (64)
    )
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_40_',
        ) # (64)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_41_',
        ) # (64)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 64, 1, 1],
                mean=0.0,
                stddev=0.17677669,
                dtype=tf.dtypes.float32,
                name='normal_42_',
            ), # (256, 64, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_43_',
        ) # (1, 1, 64, 256)
    )
    self._03ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_44_',
        ) # (256)
    )
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_45_',
        ) # (256)
    )
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_46_',
        ) # (256)
    )
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_47_',
        ) # (256)
        , trainable = False
    )
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_48_',
        ) # (256)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_49_',
            ), # (64, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_50_',
        ) # (1, 1, 256, 64)
    )
    self._04ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_51_',
        ) # (64)
    )
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_52_',
        ) # (64)
    )
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_53_',
        ) # (64)
    )
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_54_',
        ) # (64)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_55_',
        ) # (64)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[64, 64, 3, 3],
                mean=0.0,
                stddev=0.058925565,
                dtype=tf.dtypes.float32,
                name='normal_56_',
            ), # (64, 64, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_57_',
        ) # (3, 3, 64, 64)
    )
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_58_',
        ) # (64)
    )
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_59_',
        ) # (64)
    )
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='zeros_60_',
        ) # (64)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[64],
            dtype=tf.dtypes.float32,
            name='ones_61_',
        ) # (64)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 64, 1, 1],
                mean=0.0,
                stddev=0.17677669,
                dtype=tf.dtypes.float32,
                name='normal_62_',
            ), # (256, 64, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_63_',
        ) # (1, 1, 64, 256)
    )
    self._04ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_64_',
        ) # (256)
    )
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_65_',
        ) # (256)
    )
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_66_',
        ) # (256)
    )
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_67_',
        ) # (256)
        , trainable = False
    )
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_68_',
        ) # (256)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_69_',
            ), # (128, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_70_',
        ) # (1, 1, 256, 128)
    )
    self._05ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_71_',
        ) # (128)
    )
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_72_',
        ) # (128)
    )
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_73_',
        ) # (128)
    )
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_74_',
        ) # (128)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_75_',
        ) # (128)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 128, 3, 3],
                mean=0.0,
                stddev=0.041666668,
                dtype=tf.dtypes.float32,
                name='normal_76_',
            ), # (128, 128, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_77_',
        ) # (3, 3, 128, 128)
    )
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_78_',
        ) # (128)
    )
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_79_',
        ) # (128)
    )
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_80_',
        ) # (128)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_81_',
        ) # (128)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 128, 1, 1],
                mean=0.0,
                stddev=0.125,
                dtype=tf.dtypes.float32,
                name='normal_82_',
            ), # (512, 128, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_83_',
        ) # (1, 1, 128, 512)
    )
    self._05ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_84_',
        ) # (512)
    )
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_85_',
        ) # (512)
    )
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_86_',
        ) # (512)
    )
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_87_',
        ) # (512)
        , trainable = False
    )
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_88_',
        ) # (512)
        , trainable = False
    )
    self._05ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_89_',
            ), # (512, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_90_',
        ) # (1, 1, 256, 512)
    )
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_91_',
        ) # (512)
    )
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_92_',
        ) # (512)
    )
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_93_',
        ) # (512)
        , trainable = False
    )
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_94_',
        ) # (512)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_95_',
            ), # (128, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_96_',
        ) # (1, 1, 512, 128)
    )
    self._06ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_97_',
        ) # (128)
    )
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_98_',
        ) # (128)
    )
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_99_',
        ) # (128)
    )
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_100_',
        ) # (128)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_101_',
        ) # (128)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 128, 3, 3],
                mean=0.0,
                stddev=0.041666668,
                dtype=tf.dtypes.float32,
                name='normal_102_',
            ), # (128, 128, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_103_',
        ) # (3, 3, 128, 128)
    )
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_104_',
        ) # (128)
    )
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_105_',
        ) # (128)
    )
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_106_',
        ) # (128)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_107_',
        ) # (128)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 128, 1, 1],
                mean=0.0,
                stddev=0.125,
                dtype=tf.dtypes.float32,
                name='normal_108_',
            ), # (512, 128, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_109_',
        ) # (1, 1, 128, 512)
    )
    self._06ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_110_',
        ) # (512)
    )
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_111_',
        ) # (512)
    )
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_112_',
        ) # (512)
    )
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_113_',
        ) # (512)
        , trainable = False
    )
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_114_',
        ) # (512)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_115_',
            ), # (128, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_116_',
        ) # (1, 1, 512, 128)
    )
    self._07ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_117_',
        ) # (128)
    )
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_118_',
        ) # (128)
    )
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_119_',
        ) # (128)
    )
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_120_',
        ) # (128)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_121_',
        ) # (128)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 128, 3, 3],
                mean=0.0,
                stddev=0.041666668,
                dtype=tf.dtypes.float32,
                name='normal_122_',
            ), # (128, 128, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_123_',
        ) # (3, 3, 128, 128)
    )
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_124_',
        ) # (128)
    )
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_125_',
        ) # (128)
    )
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_126_',
        ) # (128)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_127_',
        ) # (128)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 128, 1, 1],
                mean=0.0,
                stddev=0.125,
                dtype=tf.dtypes.float32,
                name='normal_128_',
            ), # (512, 128, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_129_',
        ) # (1, 1, 128, 512)
    )
    self._07ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_130_',
        ) # (512)
    )
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_131_',
        ) # (512)
    )
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_132_',
        ) # (512)
    )
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_133_',
        ) # (512)
        , trainable = False
    )
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_134_',
        ) # (512)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_135_',
            ), # (128, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_136_',
        ) # (1, 1, 512, 128)
    )
    self._08ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_137_',
        ) # (128)
    )
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_138_',
        ) # (128)
    )
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_139_',
        ) # (128)
    )
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_140_',
        ) # (128)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_141_',
        ) # (128)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[128, 128, 3, 3],
                mean=0.0,
                stddev=0.041666668,
                dtype=tf.dtypes.float32,
                name='normal_142_',
            ), # (128, 128, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_143_',
        ) # (3, 3, 128, 128)
    )
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_144_',
        ) # (128)
    )
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_145_',
        ) # (128)
    )
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='zeros_146_',
        ) # (128)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[128],
            dtype=tf.dtypes.float32,
            name='ones_147_',
        ) # (128)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 128, 1, 1],
                mean=0.0,
                stddev=0.125,
                dtype=tf.dtypes.float32,
                name='normal_148_',
            ), # (512, 128, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_149_',
        ) # (1, 1, 128, 512)
    )
    self._08ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_150_',
        ) # (512)
    )
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_151_',
        ) # (512)
    )
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_152_',
        ) # (512)
    )
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_153_',
        ) # (512)
        , trainable = False
    )
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_154_',
        ) # (512)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_155_',
            ), # (256, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_156_',
        ) # (1, 1, 512, 256)
    )
    self._09ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_157_',
        ) # (256)
    )
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_158_',
        ) # (256)
    )
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_159_',
        ) # (256)
    )
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_160_',
        ) # (256)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_161_',
        ) # (256)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_162_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_163_',
        ) # (3, 3, 256, 256)
    )
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_164_',
        ) # (256)
    )
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_165_',
        ) # (256)
    )
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_166_',
        ) # (256)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_167_',
        ) # (256)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_168_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_169_',
        ) # (1, 1, 256, 1024)
    )
    self._09ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_170_',
        ) # (1024)
    )
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_171_',
        ) # (1024)
    )
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_172_',
        ) # (1024)
    )
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_173_',
        ) # (1024)
        , trainable = False
    )
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_174_',
        ) # (1024)
        , trainable = False
    )
    self._09ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_175_',
            ), # (1024, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_176_',
        ) # (1, 1, 512, 1024)
    )
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_177_',
        ) # (1024)
    )
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_178_',
        ) # (1024)
    )
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_179_',
        ) # (1024)
        , trainable = False
    )
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_180_',
        ) # (1024)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_181_',
            ), # (256, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_182_',
        ) # (1, 1, 1024, 256)
    )
    self._10ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_183_',
        ) # (256)
    )
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_184_',
        ) # (256)
    )
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_185_',
        ) # (256)
    )
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_186_',
        ) # (256)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_187_',
        ) # (256)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_188_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_189_',
        ) # (3, 3, 256, 256)
    )
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_190_',
        ) # (256)
    )
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_191_',
        ) # (256)
    )
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_192_',
        ) # (256)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_193_',
        ) # (256)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_194_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_195_',
        ) # (1, 1, 256, 1024)
    )
    self._10ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_196_',
        ) # (1024)
    )
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_197_',
        ) # (1024)
    )
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_198_',
        ) # (1024)
    )
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_199_',
        ) # (1024)
        , trainable = False
    )
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_200_',
        ) # (1024)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_201_',
            ), # (256, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_202_',
        ) # (1, 1, 1024, 256)
    )
    self._11ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_203_',
        ) # (256)
    )
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_204_',
        ) # (256)
    )
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_205_',
        ) # (256)
    )
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_206_',
        ) # (256)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_207_',
        ) # (256)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_208_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_209_',
        ) # (3, 3, 256, 256)
    )
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_210_',
        ) # (256)
    )
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_211_',
        ) # (256)
    )
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_212_',
        ) # (256)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_213_',
        ) # (256)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_214_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_215_',
        ) # (1, 1, 256, 1024)
    )
    self._11ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_216_',
        ) # (1024)
    )
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_217_',
        ) # (1024)
    )
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_218_',
        ) # (1024)
    )
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_219_',
        ) # (1024)
        , trainable = False
    )
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_220_',
        ) # (1024)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_221_',
            ), # (256, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_222_',
        ) # (1, 1, 1024, 256)
    )
    self._12ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_223_',
        ) # (256)
    )
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_224_',
        ) # (256)
    )
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_225_',
        ) # (256)
    )
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_226_',
        ) # (256)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_227_',
        ) # (256)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_228_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_229_',
        ) # (3, 3, 256, 256)
    )
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_230_',
        ) # (256)
    )
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_231_',
        ) # (256)
    )
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_232_',
        ) # (256)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_233_',
        ) # (256)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_234_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_235_',
        ) # (1, 1, 256, 1024)
    )
    self._12ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_236_',
        ) # (1024)
    )
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_237_',
        ) # (1024)
    )
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_238_',
        ) # (1024)
    )
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_239_',
        ) # (1024)
        , trainable = False
    )
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_240_',
        ) # (1024)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_241_',
            ), # (256, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_242_',
        ) # (1, 1, 1024, 256)
    )
    self._13ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_243_',
        ) # (256)
    )
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_244_',
        ) # (256)
    )
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_245_',
        ) # (256)
    )
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_246_',
        ) # (256)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_247_',
        ) # (256)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_248_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_249_',
        ) # (3, 3, 256, 256)
    )
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_250_',
        ) # (256)
    )
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_251_',
        ) # (256)
    )
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_252_',
        ) # (256)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_253_',
        ) # (256)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_254_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_255_',
        ) # (1, 1, 256, 1024)
    )
    self._13ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_256_',
        ) # (1024)
    )
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_257_',
        ) # (1024)
    )
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_258_',
        ) # (1024)
    )
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_259_',
        ) # (1024)
        , trainable = False
    )
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_260_',
        ) # (1024)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_261_',
            ), # (256, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_262_',
        ) # (1, 1, 1024, 256)
    )
    self._14ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_263_',
        ) # (256)
    )
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_264_',
        ) # (256)
    )
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_265_',
        ) # (256)
    )
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_266_',
        ) # (256)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_267_',
        ) # (256)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[256, 256, 3, 3],
                mean=0.0,
                stddev=0.029462783,
                dtype=tf.dtypes.float32,
                name='normal_268_',
            ), # (256, 256, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_269_',
        ) # (3, 3, 256, 256)
    )
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_270_',
        ) # (256)
    )
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_271_',
        ) # (256)
    )
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='zeros_272_',
        ) # (256)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[256],
            dtype=tf.dtypes.float32,
            name='ones_273_',
        ) # (256)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[1024, 256, 1, 1],
                mean=0.0,
                stddev=0.088388346,
                dtype=tf.dtypes.float32,
                name='normal_274_',
            ), # (1024, 256, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_275_',
        ) # (1, 1, 256, 1024)
    )
    self._14ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_276_',
        ) # (1024)
    )
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_277_',
        ) # (1024)
    )
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_278_',
        ) # (1024)
    )
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='zeros_279_',
        ) # (1024)
        , trainable = False
    )
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[1024],
            dtype=tf.dtypes.float32,
            name='ones_280_',
        ) # (1024)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_281_',
            ), # (512, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_282_',
        ) # (1, 1, 1024, 512)
    )
    self._15ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_283_',
        ) # (512)
    )
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_284_',
        ) # (512)
    )
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_285_',
        ) # (512)
    )
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_286_',
        ) # (512)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_287_',
        ) # (512)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 512, 3, 3],
                mean=0.0,
                stddev=0.020833334,
                dtype=tf.dtypes.float32,
                name='normal_288_',
            ), # (512, 512, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_289_',
        ) # (3, 3, 512, 512)
    )
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_290_',
        ) # (512)
    )
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_291_',
        ) # (512)
    )
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_292_',
        ) # (512)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_293_',
        ) # (512)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[2048, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_294_',
            ), # (2048, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_295_',
        ) # (1, 1, 512, 2048)
    )
    self._15ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_296_',
        ) # (2048)
    )
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_297_',
        ) # (2048)
    )
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_298_',
        ) # (2048)
    )
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_299_',
        ) # (2048)
        , trainable = False
    )
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_300_',
        ) # (2048)
        , trainable = False
    )
    self._15ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[2048, 1024, 1, 1],
                mean=0.0,
                stddev=0.044194173,
                dtype=tf.dtypes.float32,
                name='normal_301_',
            ), # (2048, 1024, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_302_',
        ) # (1, 1, 1024, 2048)
    )
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_303_',
        ) # (2048)
    )
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_304_',
        ) # (2048)
    )
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_305_',
        ) # (2048)
        , trainable = False
    )
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_306_',
        ) # (2048)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 2048, 1, 1],
                mean=0.0,
                stddev=0.03125,
                dtype=tf.dtypes.float32,
                name='normal_307_',
            ), # (512, 2048, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_308_',
        ) # (1, 1, 2048, 512)
    )
    self._16ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_309_',
        ) # (512)
    )
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_310_',
        ) # (512)
    )
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_311_',
        ) # (512)
    )
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_312_',
        ) # (512)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_313_',
        ) # (512)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 512, 3, 3],
                mean=0.0,
                stddev=0.020833334,
                dtype=tf.dtypes.float32,
                name='normal_314_',
            ), # (512, 512, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_315_',
        ) # (3, 3, 512, 512)
    )
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_316_',
        ) # (512)
    )
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_317_',
        ) # (512)
    )
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_318_',
        ) # (512)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_319_',
        ) # (512)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[2048, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_320_',
            ), # (2048, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_321_',
        ) # (1, 1, 512, 2048)
    )
    self._16ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_322_',
        ) # (2048)
    )
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_323_',
        ) # (2048)
    )
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_324_',
        ) # (2048)
    )
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_325_',
        ) # (2048)
        , trainable = False
    )
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_326_',
        ) # (2048)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 2048, 1, 1],
                mean=0.0,
                stddev=0.03125,
                dtype=tf.dtypes.float32,
                name='normal_327_',
            ), # (512, 2048, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_328_',
        ) # (1, 1, 2048, 512)
    )
    self._17ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_329_',
        ) # (512)
    )
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_330_',
        ) # (512)
    )
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_331_',
        ) # (512)
    )
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_332_',
        ) # (512)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_333_',
        ) # (512)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[512, 512, 3, 3],
                mean=0.0,
                stddev=0.020833334,
                dtype=tf.dtypes.float32,
                name='normal_334_',
            ), # (512, 512, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_335_',
        ) # (3, 3, 512, 512)
    )
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_336_',
        ) # (512)
    )
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_337_',
        ) # (512)
    )
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='zeros_338_',
        ) # (512)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[512],
            dtype=tf.dtypes.float32,
            name='ones_339_',
        ) # (512)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(
        tf.transpose(
            tf.random.normal(
                shape=[2048, 512, 1, 1],
                mean=0.0,
                stddev=0.0625,
                dtype=tf.dtypes.float32,
                name='normal_340_',
            ), # (2048, 512, 1, 1)
            perm=[2, 3, 1, 0],
            name='transpose_341_',
        ) # (1, 1, 512, 2048)
    )
    self._17ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_342_',
        ) # (2048)
    )
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_343_',
        ) # (2048)
    )
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_344_',
        ) # (2048)
    )
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(
        tf.zeros(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='zeros_345_',
        ) # (2048)
        , trainable = False
    )
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(
        tf.ones(
            shape=[2048],
            dtype=tf.dtypes.float32,
            name='ones_346_',
        ) # (2048)
        , trainable = False
    )
    self._20Linear_weight = tf.Variable(
        tf.random.normal(
            shape=[10, 2048],
            mean=0.0,
            stddev=0.03125,
            dtype=tf.dtypes.float32,
            name='normal_347_',
        ) # (10, 2048)
    )
    self._20Linear_bias = tf.Variable(
        tf.zeros(
            shape=[10],
            dtype=tf.dtypes.float32,
            name='zeros_348_',
        ) # (10)
    )

## 2
  def call(self, x):
    val1 = tf.nn.convolution(
        x, # (111, 3, 32, 32)
        filters=self._01Conv2d_weight, # (3, 3, 3, 64)
        strides=[1, 1],
        padding='SAME',
        dilations=[1, 1],
        data_format='NCHW',
        name='convolution_349_',
    ) # (111, 64, 32, 32)  
    (batchnorm1, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val1, # (111, 64, 32, 32)
                filters=self._02ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 64, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_350_',
            ), # (111, 64, 32, 32)
            bias=self._02ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_351_',
        ), # (111, 64, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._02ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_352_',
    ) # (111, 64, 32, 32)  
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm2, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm1, # (111, 64, 32, 32)
                name='relu_353_',
            ), # (111, 64, 32, 32)
            filters=self._02ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_354_',
        ), # (111, 64, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._02ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_355_',
    ) # (111, 64, 32, 32)  
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm3, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm2, # (111, 64, 32, 32)
                    name='relu_356_',
                ), # (111, 64, 32, 32)
                filters=self._02ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_357_',
            ), # (111, 256, 32, 32)
            bias=self._02ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_358_',
        ), # (111, 256, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._02ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_359_',
    ) # (111, 256, 32, 32)  
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm4, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val1, # (111, 64, 32, 32)
            filters=self._02ParallelBlock_02SequentialBlock_01Conv2d_weight, # (1, 1, 64, 256)
            strides=[1, 1],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_360_',
        ), # (111, 256, 32, 32)
        scale=self._02ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._02ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_361_',
    ) # (111, 256, 32, 32)  
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val2 = tf.nn.relu(
        tf.add(
            batchnorm3, # (111, 256, 32, 32)
            batchnorm4, # (111, 256, 32, 32)
            name='add_362_',
        ), # (111, 256, 32, 32)
        name='relu_363_',
    ) # (111, 256, 32, 32)  
    (batchnorm5, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val2, # (111, 256, 32, 32)
                filters=self._03ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 256, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_364_',
            ), # (111, 64, 32, 32)
            bias=self._03ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_365_',
        ), # (111, 64, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._03ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_366_',
    ) # (111, 64, 32, 32)  
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm6, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm5, # (111, 64, 32, 32)
                name='relu_367_',
            ), # (111, 64, 32, 32)
            filters=self._03ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_368_',
        ), # (111, 64, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._03ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_369_',
    ) # (111, 64, 32, 32)  
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm7, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm6, # (111, 64, 32, 32)
                    name='relu_370_',
                ), # (111, 64, 32, 32)
                filters=self._03ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_371_',
            ), # (111, 256, 32, 32)
            bias=self._03ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_372_',
        ), # (111, 256, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._03ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_373_',
    ) # (111, 256, 32, 32)  
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val3 = tf.nn.relu(
        tf.add(
            batchnorm7, # (111, 256, 32, 32)
            val2, # (111, 256, 32, 32)
            name='add_374_',
        ), # (111, 256, 32, 32)
        name='relu_375_',
    ) # (111, 256, 32, 32)  
    (batchnorm8, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val3, # (111, 256, 32, 32)
                filters=self._04ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 256, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_376_',
            ), # (111, 64, 32, 32)
            bias=self._04ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_377_',
        ), # (111, 64, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._04ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_378_',
    ) # (111, 64, 32, 32)  
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm9, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm8, # (111, 64, 32, 32)
                name='relu_379_',
            ), # (111, 64, 32, 32)
            filters=self._04ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_380_',
        ), # (111, 64, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._04ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_381_',
    ) # (111, 64, 32, 32)  
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm10, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm9, # (111, 64, 32, 32)
                    name='relu_382_',
                ), # (111, 64, 32, 32)
                filters=self._04ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_383_',
            ), # (111, 256, 32, 32)
            bias=self._04ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_384_',
        ), # (111, 256, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._04ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_385_',
    ) # (111, 256, 32, 32)  
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val4 = tf.nn.relu(
        tf.add(
            batchnorm10, # (111, 256, 32, 32)
            val3, # (111, 256, 32, 32)
            name='add_386_',
        ), # (111, 256, 32, 32)
        name='relu_387_',
    ) # (111, 256, 32, 32)  
    (batchnorm11, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val4, # (111, 256, 32, 32)
                filters=self._05ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 256, 128)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_388_',
            ), # (111, 128, 16, 16)
            bias=self._05ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_389_',
        ), # (111, 128, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._05ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_390_',
    ) # (111, 128, 16, 16)  
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm12, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm11, # (111, 128, 16, 16)
                name='relu_391_',
            ), # (111, 128, 16, 16)
            filters=self._05ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_392_',
        ), # (111, 128, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._05ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_393_',
    ) # (111, 128, 16, 16)  
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm13, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm12, # (111, 128, 16, 16)
                    name='relu_394_',
                ), # (111, 128, 16, 16)
                filters=self._05ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_395_',
            ), # (111, 512, 16, 16)
            bias=self._05ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_396_',
        ), # (111, 512, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._05ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_397_',
    ) # (111, 512, 16, 16)  
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm14, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val4, # (111, 256, 32, 32)
            filters=self._05ParallelBlock_02SequentialBlock_01Conv2d_weight, # (1, 1, 256, 512)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_398_',
        ), # (111, 512, 16, 16)
        scale=self._05ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._05ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_399_',
    ) # (111, 512, 16, 16)  
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val5 = tf.nn.relu(
        tf.add(
            batchnorm13, # (111, 512, 16, 16)
            batchnorm14, # (111, 512, 16, 16)
            name='add_400_',
        ), # (111, 512, 16, 16)
        name='relu_401_',
    ) # (111, 512, 16, 16)  
    (batchnorm15, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val5, # (111, 512, 16, 16)
                filters=self._06ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_402_',
            ), # (111, 128, 16, 16)
            bias=self._06ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_403_',
        ), # (111, 128, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._06ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_404_',
    ) # (111, 128, 16, 16)  
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm16, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm15, # (111, 128, 16, 16)
                name='relu_405_',
            ), # (111, 128, 16, 16)
            filters=self._06ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_406_',
        ), # (111, 128, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._06ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_407_',
    ) # (111, 128, 16, 16)  
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm17, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm16, # (111, 128, 16, 16)
                    name='relu_408_',
                ), # (111, 128, 16, 16)
                filters=self._06ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_409_',
            ), # (111, 512, 16, 16)
            bias=self._06ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_410_',
        ), # (111, 512, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._06ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_411_',
    ) # (111, 512, 16, 16)  
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val6 = tf.nn.relu(
        tf.add(
            batchnorm17, # (111, 512, 16, 16)
            val5, # (111, 512, 16, 16)
            name='add_412_',
        ), # (111, 512, 16, 16)
        name='relu_413_',
    ) # (111, 512, 16, 16)  
    (batchnorm18, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val6, # (111, 512, 16, 16)
                filters=self._07ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_414_',
            ), # (111, 128, 16, 16)
            bias=self._07ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_415_',
        ), # (111, 128, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._07ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_416_',
    ) # (111, 128, 16, 16)  
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm19, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm18, # (111, 128, 16, 16)
                name='relu_417_',
            ), # (111, 128, 16, 16)
            filters=self._07ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_418_',
        ), # (111, 128, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._07ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_419_',
    ) # (111, 128, 16, 16)  
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm20, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm19, # (111, 128, 16, 16)
                    name='relu_420_',
                ), # (111, 128, 16, 16)
                filters=self._07ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_421_',
            ), # (111, 512, 16, 16)
            bias=self._07ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_422_',
        ), # (111, 512, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._07ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_423_',
    ) # (111, 512, 16, 16)  
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val7 = tf.nn.relu(
        tf.add(
            batchnorm20, # (111, 512, 16, 16)
            val6, # (111, 512, 16, 16)
            name='add_424_',
        ), # (111, 512, 16, 16)
        name='relu_425_',
    ) # (111, 512, 16, 16)  
    (batchnorm21, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val7, # (111, 512, 16, 16)
                filters=self._08ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_426_',
            ), # (111, 128, 16, 16)
            bias=self._08ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_427_',
        ), # (111, 128, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._08ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_428_',
    ) # (111, 128, 16, 16)  
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm22, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm21, # (111, 128, 16, 16)
                name='relu_429_',
            ), # (111, 128, 16, 16)
            filters=self._08ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_430_',
        ), # (111, 128, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._08ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_431_',
    ) # (111, 128, 16, 16)  
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm23, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm22, # (111, 128, 16, 16)
                    name='relu_432_',
                ), # (111, 128, 16, 16)
                filters=self._08ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_433_',
            ), # (111, 512, 16, 16)
            bias=self._08ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_434_',
        ), # (111, 512, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._08ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_435_',
    ) # (111, 512, 16, 16)  
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val8 = tf.nn.relu(
        tf.add(
            batchnorm23, # (111, 512, 16, 16)
            val7, # (111, 512, 16, 16)
            name='add_436_',
        ), # (111, 512, 16, 16)
        name='relu_437_',
    ) # (111, 512, 16, 16)  
    (batchnorm24, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val8, # (111, 512, 16, 16)
                filters=self._09ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 512, 256)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_438_',
            ), # (111, 256, 8, 8)
            bias=self._09ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_439_',
        ), # (111, 256, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._09ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_440_',
    ) # (111, 256, 8, 8)  
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm25, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm24, # (111, 256, 8, 8)
                name='relu_441_',
            ), # (111, 256, 8, 8)
            filters=self._09ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_442_',
        ), # (111, 256, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._09ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_443_',
    ) # (111, 256, 8, 8)  
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm26, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm25, # (111, 256, 8, 8)
                    name='relu_444_',
                ), # (111, 256, 8, 8)
                filters=self._09ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_445_',
            ), # (111, 1024, 8, 8)
            bias=self._09ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_446_',
        ), # (111, 1024, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._09ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_447_',
    ) # (111, 1024, 8, 8)  
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm27, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val8, # (111, 512, 16, 16)
            filters=self._09ParallelBlock_02SequentialBlock_01Conv2d_weight, # (1, 1, 512, 1024)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_448_',
        ), # (111, 1024, 8, 8)
        scale=self._09ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (1024)
        offset=self._09ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (1024)
        mean=self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (1024)
        variance=self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_449_',
    ) # (111, 1024, 8, 8)  
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val9 = tf.nn.relu(
        tf.add(
            batchnorm26, # (111, 1024, 8, 8)
            batchnorm27, # (111, 1024, 8, 8)
            name='add_450_',
        ), # (111, 1024, 8, 8)
        name='relu_451_',
    ) # (111, 1024, 8, 8)  
    (batchnorm28, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val9, # (111, 1024, 8, 8)
                filters=self._10ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_452_',
            ), # (111, 256, 8, 8)
            bias=self._10ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_453_',
        ), # (111, 256, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._10ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_454_',
    ) # (111, 256, 8, 8)  
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm29, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm28, # (111, 256, 8, 8)
                name='relu_455_',
            ), # (111, 256, 8, 8)
            filters=self._10ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_456_',
        ), # (111, 256, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._10ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_457_',
    ) # (111, 256, 8, 8)  
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm30, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm29, # (111, 256, 8, 8)
                    name='relu_458_',
                ), # (111, 256, 8, 8)
                filters=self._10ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_459_',
            ), # (111, 1024, 8, 8)
            bias=self._10ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_460_',
        ), # (111, 1024, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._10ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_461_',
    ) # (111, 1024, 8, 8)  
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val10 = tf.nn.relu(
        tf.add(
            batchnorm30, # (111, 1024, 8, 8)
            val9, # (111, 1024, 8, 8)
            name='add_462_',
        ), # (111, 1024, 8, 8)
        name='relu_463_',
    ) # (111, 1024, 8, 8)  
    (batchnorm31, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val10, # (111, 1024, 8, 8)
                filters=self._11ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_464_',
            ), # (111, 256, 8, 8)
            bias=self._11ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_465_',
        ), # (111, 256, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._11ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_466_',
    ) # (111, 256, 8, 8)  
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm32, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm31, # (111, 256, 8, 8)
                name='relu_467_',
            ), # (111, 256, 8, 8)
            filters=self._11ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_468_',
        ), # (111, 256, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._11ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_469_',
    ) # (111, 256, 8, 8)  
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm33, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm32, # (111, 256, 8, 8)
                    name='relu_470_',
                ), # (111, 256, 8, 8)
                filters=self._11ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_471_',
            ), # (111, 1024, 8, 8)
            bias=self._11ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_472_',
        ), # (111, 1024, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._11ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_473_',
    ) # (111, 1024, 8, 8)  
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val11 = tf.nn.relu(
        tf.add(
            batchnorm33, # (111, 1024, 8, 8)
            val10, # (111, 1024, 8, 8)
            name='add_474_',
        ), # (111, 1024, 8, 8)
        name='relu_475_',
    ) # (111, 1024, 8, 8)  
    (batchnorm34, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val11, # (111, 1024, 8, 8)
                filters=self._12ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_476_',
            ), # (111, 256, 8, 8)
            bias=self._12ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_477_',
        ), # (111, 256, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._12ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_478_',
    ) # (111, 256, 8, 8)  
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm35, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm34, # (111, 256, 8, 8)
                name='relu_479_',
            ), # (111, 256, 8, 8)
            filters=self._12ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_480_',
        ), # (111, 256, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._12ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_481_',
    ) # (111, 256, 8, 8)  
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm36, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm35, # (111, 256, 8, 8)
                    name='relu_482_',
                ), # (111, 256, 8, 8)
                filters=self._12ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_483_',
            ), # (111, 1024, 8, 8)
            bias=self._12ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_484_',
        ), # (111, 1024, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._12ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_485_',
    ) # (111, 1024, 8, 8)  
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val12 = tf.nn.relu(
        tf.add(
            batchnorm36, # (111, 1024, 8, 8)
            val11, # (111, 1024, 8, 8)
            name='add_486_',
        ), # (111, 1024, 8, 8)
        name='relu_487_',
    ) # (111, 1024, 8, 8)  
    (batchnorm37, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val12, # (111, 1024, 8, 8)
                filters=self._13ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_488_',
            ), # (111, 256, 8, 8)
            bias=self._13ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_489_',
        ), # (111, 256, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._13ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_490_',
    ) # (111, 256, 8, 8)  
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm38, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm37, # (111, 256, 8, 8)
                name='relu_491_',
            ), # (111, 256, 8, 8)
            filters=self._13ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_492_',
        ), # (111, 256, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._13ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_493_',
    ) # (111, 256, 8, 8)  
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm39, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm38, # (111, 256, 8, 8)
                    name='relu_494_',
                ), # (111, 256, 8, 8)
                filters=self._13ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_495_',
            ), # (111, 1024, 8, 8)
            bias=self._13ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_496_',
        ), # (111, 1024, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._13ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_497_',
    ) # (111, 1024, 8, 8)  
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val13 = tf.nn.relu(
        tf.add(
            batchnorm39, # (111, 1024, 8, 8)
            val12, # (111, 1024, 8, 8)
            name='add_498_',
        ), # (111, 1024, 8, 8)
        name='relu_499_',
    ) # (111, 1024, 8, 8)  
    (batchnorm40, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val13, # (111, 1024, 8, 8)
                filters=self._14ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_500_',
            ), # (111, 256, 8, 8)
            bias=self._14ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_501_',
        ), # (111, 256, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._14ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_502_',
    ) # (111, 256, 8, 8)  
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm41, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm40, # (111, 256, 8, 8)
                name='relu_503_',
            ), # (111, 256, 8, 8)
            filters=self._14ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_504_',
        ), # (111, 256, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._14ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_505_',
    ) # (111, 256, 8, 8)  
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm42, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm41, # (111, 256, 8, 8)
                    name='relu_506_',
                ), # (111, 256, 8, 8)
                filters=self._14ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_507_',
            ), # (111, 1024, 8, 8)
            bias=self._14ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_508_',
        ), # (111, 1024, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._14ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_509_',
    ) # (111, 1024, 8, 8)  
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val14 = tf.nn.relu(
        tf.add(
            batchnorm42, # (111, 1024, 8, 8)
            val13, # (111, 1024, 8, 8)
            name='add_510_',
        ), # (111, 1024, 8, 8)
        name='relu_511_',
    ) # (111, 1024, 8, 8)  
    (batchnorm43, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val14, # (111, 1024, 8, 8)
                filters=self._15ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 512)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_512_',
            ), # (111, 512, 4, 4)
            bias=self._15ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_513_',
        ), # (111, 512, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._15ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_514_',
    ) # (111, 512, 4, 4)  
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm44, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm43, # (111, 512, 4, 4)
                name='relu_515_',
            ), # (111, 512, 4, 4)
            filters=self._15ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_516_',
        ), # (111, 512, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._15ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_517_',
    ) # (111, 512, 4, 4)  
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm45, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm44, # (111, 512, 4, 4)
                    name='relu_518_',
                ), # (111, 512, 4, 4)
                filters=self._15ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 512, 2048)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_519_',
            ), # (111, 2048, 4, 4)
            bias=self._15ParallelBlock_01SequentialBlock_07Conv2d_bias, # (2048)
            data_format='NCHW',
            name='bias_add_520_',
        ), # (111, 2048, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (2048)
        offset=self._15ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (2048)
        mean=self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (2048)
        variance=self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_521_',
    ) # (111, 2048, 4, 4)  
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm46, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val14, # (111, 1024, 8, 8)
            filters=self._15ParallelBlock_02SequentialBlock_01Conv2d_weight, # (1, 1, 1024, 2048)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_522_',
        ), # (111, 2048, 4, 4)
        scale=self._15ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (2048)
        offset=self._15ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (2048)
        mean=self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (2048)
        variance=self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_523_',
    ) # (111, 2048, 4, 4)  
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val15 = tf.nn.relu(
        tf.add(
            batchnorm45, # (111, 2048, 4, 4)
            batchnorm46, # (111, 2048, 4, 4)
            name='add_524_',
        ), # (111, 2048, 4, 4)
        name='relu_525_',
    ) # (111, 2048, 4, 4)  
    (batchnorm47, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val15, # (111, 2048, 4, 4)
                filters=self._16ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 2048, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_526_',
            ), # (111, 512, 4, 4)
            bias=self._16ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_527_',
        ), # (111, 512, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._16ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_528_',
    ) # (111, 512, 4, 4)  
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm48, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm47, # (111, 512, 4, 4)
                name='relu_529_',
            ), # (111, 512, 4, 4)
            filters=self._16ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_530_',
        ), # (111, 512, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._16ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_531_',
    ) # (111, 512, 4, 4)  
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm49, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm48, # (111, 512, 4, 4)
                    name='relu_532_',
                ), # (111, 512, 4, 4)
                filters=self._16ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 512, 2048)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_533_',
            ), # (111, 2048, 4, 4)
            bias=self._16ParallelBlock_01SequentialBlock_07Conv2d_bias, # (2048)
            data_format='NCHW',
            name='bias_add_534_',
        ), # (111, 2048, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (2048)
        offset=self._16ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (2048)
        mean=self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (2048)
        variance=self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_535_',
    ) # (111, 2048, 4, 4)  
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val16 = tf.nn.relu(
        tf.add(
            batchnorm49, # (111, 2048, 4, 4)
            val15, # (111, 2048, 4, 4)
            name='add_536_',
        ), # (111, 2048, 4, 4)
        name='relu_537_',
    ) # (111, 2048, 4, 4)  
    (batchnorm50, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val16, # (111, 2048, 4, 4)
                filters=self._17ParallelBlock_01SequentialBlock_01Conv2d_weight, # (1, 1, 2048, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_538_',
            ), # (111, 512, 4, 4)
            bias=self._17ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_539_',
        ), # (111, 512, 4, 4)
        scale=self._17ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._17ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_540_',
    ) # (111, 512, 4, 4)  
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm51, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm50, # (111, 512, 4, 4)
                name='relu_541_',
            ), # (111, 512, 4, 4)
            filters=self._17ParallelBlock_01SequentialBlock_04Conv2d_weight, # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_542_',
        ), # (111, 512, 4, 4)
        scale=self._17ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._17ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_543_',
    ) # (111, 512, 4, 4)  
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm52, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm51, # (111, 512, 4, 4)
                    name='relu_544_',
                ), # (111, 512, 4, 4)
                filters=self._17ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1, 1, 512, 2048)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_545_',
            ), # (111, 2048, 4, 4)
            bias=self._17ParallelBlock_01SequentialBlock_07Conv2d_bias, # (2048)
            data_format='NCHW',
            name='bias_add_546_',
        ), # (111, 2048, 4, 4)
        scale=self._17ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (2048)
        offset=self._17ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (2048)
        mean=self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (2048)
        variance=self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_547_',
    ) # (111, 2048, 4, 4)  
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    result = tf.reshape(
        tf.nn.bias_add(
            tf.matmul(
                tf.reshape(
                    tf.reshape(
                        tf.reduce_mean(
                            tf.nn.relu(
                                tf.add(
                                    batchnorm52, # (111, 2048, 4, 4)
                                    val16, # (111, 2048, 4, 4)
                                    name='add_548_',
                                ), # (111, 2048, 4, 4)
                                name='relu_549_',
                            ), # (111, 2048, 4, 4)
                            axis=[2, 3],
                            name='reduce_mean_550_',
                        ), # (111, 2048, 1, 1)
                        shape=[-1, 2048],
                        name='reshape_551_',
                    ), # (111, 2048)
                    shape=[-1, 2048],
                    name='reshape_552_',
                ), # (111, 2048)
                b=self._20Linear_weight, # (10, 2048)
                transpose_b=True,
                name='matmul_553_',
            ), # (111, 10)
            bias=self._20Linear_bias, # (10)
            data_format=None,
            name='bias_add_554_',
        ), # (111, 10)
        shape=[-1, 10],
        name='reshape_555_',
    ) # (111, 10)
    return result

## 2
def loss(label, prediction):
    result = tf.reduce_mean(
        tf.negative(
            tf.gather(
                tf.nn.log_softmax(
                    prediction, # (111, 10)
                    axis=-1,
                    name='log_softmax_556_',
                ), # (111, 10)
                indices=label, # (111)
                batch_dims=1,
                name='gather_557_',
            ), # (111, 1)
            name='negative_558_',
        ), # (111, 1)
        name='reduce_mean_559_',
    ) # ()
    return result

# number of epochs was 2
# number of prediction functions is 1
# number of loss functions is 1

