class MyModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 3, 3, 3],
        mean=0.0,
        stddev=0.27216554,
        dtype=tf.dtypes.float32,
        name='normal_1_',
    ))
    self._02ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 64, 1, 1],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_2_',
    ))
    self._02ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_3_',
    ))
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_4_',
    ))
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_5_',
    ))
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_6_',
    ), trainable = False)
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_7_',
    ), trainable = False)
    self._02ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 64, 3, 3],
        mean=0.0,
        stddev=0.058925565,
        dtype=tf.dtypes.float32,
        name='normal_8_',
    ))
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_9_',
    ))
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_10_',
    ))
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_11_',
    ), trainable = False)
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_12_',
    ), trainable = False)
    self._02ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 64, 1, 1],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_13_',
    ))
    self._02ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_14_',
    ))
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_15_',
    ))
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_16_',
    ))
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_17_',
    ), trainable = False)
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_18_',
    ), trainable = False)
    self._02ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 64, 1, 1],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_19_',
    ))
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_20_',
    ))
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_21_',
    ))
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_22_',
    ), trainable = False)
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_23_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_24_',
    ))
    self._03ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_25_',
    ))
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_26_',
    ))
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_27_',
    ))
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_28_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_29_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 64, 3, 3],
        mean=0.0,
        stddev=0.058925565,
        dtype=tf.dtypes.float32,
        name='normal_30_',
    ))
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_31_',
    ))
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_32_',
    ))
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_33_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_34_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 64, 1, 1],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_35_',
    ))
    self._03ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_36_',
    ))
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_37_',
    ))
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_38_',
    ))
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_39_',
    ), trainable = False)
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_40_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_41_',
    ))
    self._04ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_42_',
    ))
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_43_',
    ))
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_44_',
    ))
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_45_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_46_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[64, 64, 3, 3],
        mean=0.0,
        stddev=0.058925565,
        dtype=tf.dtypes.float32,
        name='normal_47_',
    ))
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_48_',
    ))
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_49_',
    ))
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='zeros_50_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[64],
        dtype=tf.dtypes.float32,
        name='ones_51_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 64, 1, 1],
        mean=0.0,
        stddev=0.17677669,
        dtype=tf.dtypes.float32,
        name='normal_52_',
    ))
    self._04ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_53_',
    ))
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_54_',
    ))
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_55_',
    ))
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_56_',
    ), trainable = False)
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_57_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_58_',
    ))
    self._05ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_59_',
    ))
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_60_',
    ))
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_61_',
    ))
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_62_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_63_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 128, 3, 3],
        mean=0.0,
        stddev=0.041666668,
        dtype=tf.dtypes.float32,
        name='normal_64_',
    ))
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_65_',
    ))
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_66_',
    ))
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_67_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_68_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 128, 1, 1],
        mean=0.0,
        stddev=0.125,
        dtype=tf.dtypes.float32,
        name='normal_69_',
    ))
    self._05ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_70_',
    ))
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_71_',
    ))
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_72_',
    ))
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_73_',
    ), trainable = False)
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_74_',
    ), trainable = False)
    self._05ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_75_',
    ))
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_76_',
    ))
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_77_',
    ))
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_78_',
    ), trainable = False)
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_79_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_80_',
    ))
    self._06ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_81_',
    ))
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_82_',
    ))
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_83_',
    ))
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_84_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_85_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 128, 3, 3],
        mean=0.0,
        stddev=0.041666668,
        dtype=tf.dtypes.float32,
        name='normal_86_',
    ))
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_87_',
    ))
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_88_',
    ))
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_89_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_90_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 128, 1, 1],
        mean=0.0,
        stddev=0.125,
        dtype=tf.dtypes.float32,
        name='normal_91_',
    ))
    self._06ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_92_',
    ))
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_93_',
    ))
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_94_',
    ))
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_95_',
    ), trainable = False)
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_96_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_97_',
    ))
    self._07ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_98_',
    ))
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_99_',
    ))
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_100_',
    ))
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_101_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_102_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 128, 3, 3],
        mean=0.0,
        stddev=0.041666668,
        dtype=tf.dtypes.float32,
        name='normal_103_',
    ))
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_104_',
    ))
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_105_',
    ))
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_106_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_107_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 128, 1, 1],
        mean=0.0,
        stddev=0.125,
        dtype=tf.dtypes.float32,
        name='normal_108_',
    ))
    self._07ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_109_',
    ))
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_110_',
    ))
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_111_',
    ))
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_112_',
    ), trainable = False)
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_113_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_114_',
    ))
    self._08ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_115_',
    ))
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_116_',
    ))
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_117_',
    ))
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_118_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_119_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[128, 128, 3, 3],
        mean=0.0,
        stddev=0.041666668,
        dtype=tf.dtypes.float32,
        name='normal_120_',
    ))
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_121_',
    ))
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_122_',
    ))
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='zeros_123_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[128],
        dtype=tf.dtypes.float32,
        name='ones_124_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 128, 1, 1],
        mean=0.0,
        stddev=0.125,
        dtype=tf.dtypes.float32,
        name='normal_125_',
    ))
    self._08ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_126_',
    ))
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_127_',
    ))
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_128_',
    ))
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_129_',
    ), trainable = False)
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_130_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_131_',
    ))
    self._09ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_132_',
    ))
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_133_',
    ))
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_134_',
    ))
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_135_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_136_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_137_',
    ))
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_138_',
    ))
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_139_',
    ))
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_140_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_141_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_142_',
    ))
    self._09ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_143_',
    ))
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_144_',
    ))
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_145_',
    ))
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_146_',
    ), trainable = False)
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_147_',
    ), trainable = False)
    self._09ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_148_',
    ))
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_149_',
    ))
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_150_',
    ))
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_151_',
    ), trainable = False)
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_152_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_153_',
    ))
    self._10ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_154_',
    ))
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_155_',
    ))
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_156_',
    ))
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_157_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_158_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_159_',
    ))
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_160_',
    ))
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_161_',
    ))
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_162_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_163_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_164_',
    ))
    self._10ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_165_',
    ))
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_166_',
    ))
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_167_',
    ))
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_168_',
    ), trainable = False)
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_169_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_170_',
    ))
    self._11ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_171_',
    ))
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_172_',
    ))
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_173_',
    ))
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_174_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_175_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_176_',
    ))
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_177_',
    ))
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_178_',
    ))
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_179_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_180_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_181_',
    ))
    self._11ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_182_',
    ))
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_183_',
    ))
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_184_',
    ))
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_185_',
    ), trainable = False)
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_186_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_187_',
    ))
    self._12ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_188_',
    ))
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_189_',
    ))
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_190_',
    ))
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_191_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_192_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_193_',
    ))
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_194_',
    ))
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_195_',
    ))
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_196_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_197_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_198_',
    ))
    self._12ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_199_',
    ))
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_200_',
    ))
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_201_',
    ))
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_202_',
    ), trainable = False)
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_203_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_204_',
    ))
    self._13ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_205_',
    ))
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_206_',
    ))
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_207_',
    ))
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_208_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_209_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_210_',
    ))
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_211_',
    ))
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_212_',
    ))
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_213_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_214_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_215_',
    ))
    self._13ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_216_',
    ))
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_217_',
    ))
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_218_',
    ))
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_219_',
    ), trainable = False)
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_220_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_221_',
    ))
    self._14ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_222_',
    ))
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_223_',
    ))
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_224_',
    ))
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_225_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_226_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[256, 256, 3, 3],
        mean=0.0,
        stddev=0.029462783,
        dtype=tf.dtypes.float32,
        name='normal_227_',
    ))
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_228_',
    ))
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_229_',
    ))
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='zeros_230_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[256],
        dtype=tf.dtypes.float32,
        name='ones_231_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[1024, 256, 1, 1],
        mean=0.0,
        stddev=0.088388346,
        dtype=tf.dtypes.float32,
        name='normal_232_',
    ))
    self._14ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_233_',
    ))
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_234_',
    ))
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_235_',
    ))
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='zeros_236_',
    ), trainable = False)
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[1024],
        dtype=tf.dtypes.float32,
        name='ones_237_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_238_',
    ))
    self._15ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_239_',
    ))
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_240_',
    ))
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_241_',
    ))
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_242_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_243_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 512, 3, 3],
        mean=0.0,
        stddev=0.020833334,
        dtype=tf.dtypes.float32,
        name='normal_244_',
    ))
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_245_',
    ))
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_246_',
    ))
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_247_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_248_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[2048, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_249_',
    ))
    self._15ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_250_',
    ))
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_251_',
    ))
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_252_',
    ))
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_253_',
    ), trainable = False)
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_254_',
    ), trainable = False)
    self._15ParallelBlock_02SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[2048, 1024, 1, 1],
        mean=0.0,
        stddev=0.044194173,
        dtype=tf.dtypes.float32,
        name='normal_255_',
    ))
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_256_',
    ))
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_257_',
    ))
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_258_',
    ), trainable = False)
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_259_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 2048, 1, 1],
        mean=0.0,
        stddev=0.03125,
        dtype=tf.dtypes.float32,
        name='normal_260_',
    ))
    self._16ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_261_',
    ))
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_262_',
    ))
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_263_',
    ))
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_264_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_265_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 512, 3, 3],
        mean=0.0,
        stddev=0.020833334,
        dtype=tf.dtypes.float32,
        name='normal_266_',
    ))
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_267_',
    ))
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_268_',
    ))
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_269_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_270_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[2048, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_271_',
    ))
    self._16ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_272_',
    ))
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_273_',
    ))
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_274_',
    ))
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_275_',
    ), trainable = False)
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_276_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_01Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 2048, 1, 1],
        mean=0.0,
        stddev=0.03125,
        dtype=tf.dtypes.float32,
        name='normal_277_',
    ))
    self._17ParallelBlock_01SequentialBlock_01Conv2d_bias = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_278_',
    ))
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_279_',
    ))
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_280_',
    ))
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_281_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_282_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_04Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[512, 512, 3, 3],
        mean=0.0,
        stddev=0.020833334,
        dtype=tf.dtypes.float32,
        name='normal_283_',
    ))
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_284_',
    ))
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_285_',
    ))
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='zeros_286_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[512],
        dtype=tf.dtypes.float32,
        name='ones_287_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_07Conv2d_weight = tf.Variable(tf.random.normal(
        shape=[2048, 512, 1, 1],
        mean=0.0,
        stddev=0.0625,
        dtype=tf.dtypes.float32,
        name='normal_288_',
    ))
    self._17ParallelBlock_01SequentialBlock_07Conv2d_bias = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_289_',
    ))
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_gamma = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_290_',
    ))
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_beta = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_291_',
    ))
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningMean = tf.Variable(tf.zeros(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='zeros_292_',
    ), trainable = False)
    self._17ParallelBlock_01SequentialBlock_08BatchNorm_runningVar = tf.Variable(tf.ones(
        shape=[2048],
        dtype=tf.dtypes.float32,
        name='ones_293_',
    ), trainable = False)
    self._20Linear_weight = tf.Variable(tf.random.normal(
        shape=[10, 2048],
        mean=0.0,
        stddev=0.03125,
        dtype=tf.dtypes.float32,
        name='normal_294_',
    ))
    self._20Linear_bias = tf.Variable(tf.zeros(
        shape=[10],
        dtype=tf.dtypes.float32,
        name='zeros_295_',
    ))

## 1
  def call(self, x):
    val1 = tf.nn.convolution(
        x, # (111, 3, 32, 32)
        filters=tf.transpose(
            self._01Conv2d_weight, # (64, 3, 3, 3)
            perm=[2, 3, 1, 0],
            name='transpose_296_',
        ), # (3, 3, 3, 64)
        strides=[1, 1],
        padding='SAME',
        dilations=[1, 1],
        data_format='NCHW',
        name='convolution_297_',
    )  
    (batchnorm1, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val1, # (111, 64, 32, 32)
                filters=tf.transpose(
                    self._02ParallelBlock_01SequentialBlock_01Conv2d_weight, # (64, 64, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_298_',
                ), # (1, 1, 64, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_299_',
            ), # (111, 64, 32, 32)
            bias=self._02ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_300_',
        ), # (111, 64, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._02ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_301_',
    )  
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm2, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm1, # (111, 64, 32, 32)
                name='relu_303_',
            ), # (111, 64, 32, 32)
            filters=tf.transpose(
                self._02ParallelBlock_01SequentialBlock_04Conv2d_weight, # (64, 64, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_302_',
            ), # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_304_',
        ), # (111, 64, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._02ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_305_',
    )  
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm3, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm2, # (111, 64, 32, 32)
                    name='relu_307_',
                ), # (111, 64, 32, 32)
                filters=tf.transpose(
                    self._02ParallelBlock_01SequentialBlock_07Conv2d_weight, # (256, 64, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_306_',
                ), # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_308_',
            ), # (111, 256, 32, 32)
            bias=self._02ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_309_',
        ), # (111, 256, 32, 32)
        scale=self._02ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._02ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_310_',
    )  
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm4, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val1, # (111, 64, 32, 32)
            filters=tf.transpose(
                self._02ParallelBlock_02SequentialBlock_01Conv2d_weight, # (256, 64, 1, 1)
                perm=[2, 3, 1, 0],
                name='transpose_311_',
            ), # (1, 1, 64, 256)
            strides=[1, 1],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_312_',
        ), # (111, 256, 32, 32)
        scale=self._02ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._02ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_313_',
    )  
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._02ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val2 = tf.nn.relu(
        tf.add(
            batchnorm3, # (111, 256, 32, 32)
            batchnorm4, # (111, 256, 32, 32)
            name='add_314_',
        ), # (111, 256, 32, 32)
        name='relu_315_',
    )  
    (batchnorm5, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val2, # (111, 256, 32, 32)
                filters=tf.transpose(
                    self._03ParallelBlock_01SequentialBlock_01Conv2d_weight, # (64, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_316_',
                ), # (1, 1, 256, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_317_',
            ), # (111, 64, 32, 32)
            bias=self._03ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_318_',
        ), # (111, 64, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._03ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_319_',
    )  
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm6, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm5, # (111, 64, 32, 32)
                name='relu_321_',
            ), # (111, 64, 32, 32)
            filters=tf.transpose(
                self._03ParallelBlock_01SequentialBlock_04Conv2d_weight, # (64, 64, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_320_',
            ), # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_322_',
        ), # (111, 64, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._03ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_323_',
    )  
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm7, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm6, # (111, 64, 32, 32)
                    name='relu_325_',
                ), # (111, 64, 32, 32)
                filters=tf.transpose(
                    self._03ParallelBlock_01SequentialBlock_07Conv2d_weight, # (256, 64, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_324_',
                ), # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_326_',
            ), # (111, 256, 32, 32)
            bias=self._03ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_327_',
        ), # (111, 256, 32, 32)
        scale=self._03ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._03ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_328_',
    )  
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._03ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val3 = tf.nn.relu(
        tf.add(
            batchnorm7, # (111, 256, 32, 32)
            val2, # (111, 256, 32, 32)
            name='add_329_',
        ), # (111, 256, 32, 32)
        name='relu_330_',
    )  
    (batchnorm8, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val3, # (111, 256, 32, 32)
                filters=tf.transpose(
                    self._04ParallelBlock_01SequentialBlock_01Conv2d_weight, # (64, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_331_',
                ), # (1, 1, 256, 64)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_332_',
            ), # (111, 64, 32, 32)
            bias=self._04ParallelBlock_01SequentialBlock_01Conv2d_bias, # (64)
            data_format='NCHW',
            name='bias_add_333_',
        ), # (111, 64, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (64)
        offset=self._04ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (64)
        mean=self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (64)
        variance=self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (64)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_334_',
    )  
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm9, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm8, # (111, 64, 32, 32)
                name='relu_336_',
            ), # (111, 64, 32, 32)
            filters=tf.transpose(
                self._04ParallelBlock_01SequentialBlock_04Conv2d_weight, # (64, 64, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_335_',
            ), # (3, 3, 64, 64)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_337_',
        ), # (111, 64, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (64)
        offset=self._04ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (64)
        mean=self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (64)
        variance=self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (64)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_338_',
    )  
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm10, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm9, # (111, 64, 32, 32)
                    name='relu_340_',
                ), # (111, 64, 32, 32)
                filters=tf.transpose(
                    self._04ParallelBlock_01SequentialBlock_07Conv2d_weight, # (256, 64, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_339_',
                ), # (1, 1, 64, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_341_',
            ), # (111, 256, 32, 32)
            bias=self._04ParallelBlock_01SequentialBlock_07Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_342_',
        ), # (111, 256, 32, 32)
        scale=self._04ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (256)
        offset=self._04ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (256)
        mean=self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (256)
        variance=self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_343_',
    )  
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._04ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val4 = tf.nn.relu(
        tf.add(
            batchnorm10, # (111, 256, 32, 32)
            val3, # (111, 256, 32, 32)
            name='add_344_',
        ), # (111, 256, 32, 32)
        name='relu_345_',
    )  
    (batchnorm11, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val4, # (111, 256, 32, 32)
                filters=tf.transpose(
                    self._05ParallelBlock_01SequentialBlock_01Conv2d_weight, # (128, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_346_',
                ), # (1, 1, 256, 128)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_347_',
            ), # (111, 128, 16, 16)
            bias=self._05ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_348_',
        ), # (111, 128, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._05ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_349_',
    )  
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm12, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm11, # (111, 128, 16, 16)
                name='relu_351_',
            ), # (111, 128, 16, 16)
            filters=tf.transpose(
                self._05ParallelBlock_01SequentialBlock_04Conv2d_weight, # (128, 128, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_350_',
            ), # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_352_',
        ), # (111, 128, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._05ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_353_',
    )  
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm13, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm12, # (111, 128, 16, 16)
                    name='relu_355_',
                ), # (111, 128, 16, 16)
                filters=tf.transpose(
                    self._05ParallelBlock_01SequentialBlock_07Conv2d_weight, # (512, 128, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_354_',
                ), # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_356_',
            ), # (111, 512, 16, 16)
            bias=self._05ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_357_',
        ), # (111, 512, 16, 16)
        scale=self._05ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._05ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_358_',
    )  
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm14, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val4, # (111, 256, 32, 32)
            filters=tf.transpose(
                self._05ParallelBlock_02SequentialBlock_01Conv2d_weight, # (512, 256, 1, 1)
                perm=[2, 3, 1, 0],
                name='transpose_359_',
            ), # (1, 1, 256, 512)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_360_',
        ), # (111, 512, 16, 16)
        scale=self._05ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._05ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_361_',
    )  
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._05ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val5 = tf.nn.relu(
        tf.add(
            batchnorm13, # (111, 512, 16, 16)
            batchnorm14, # (111, 512, 16, 16)
            name='add_362_',
        ), # (111, 512, 16, 16)
        name='relu_363_',
    )  
    (batchnorm15, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val5, # (111, 512, 16, 16)
                filters=tf.transpose(
                    self._06ParallelBlock_01SequentialBlock_01Conv2d_weight, # (128, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_364_',
                ), # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_365_',
            ), # (111, 128, 16, 16)
            bias=self._06ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_366_',
        ), # (111, 128, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._06ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_367_',
    )  
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm16, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm15, # (111, 128, 16, 16)
                name='relu_369_',
            ), # (111, 128, 16, 16)
            filters=tf.transpose(
                self._06ParallelBlock_01SequentialBlock_04Conv2d_weight, # (128, 128, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_368_',
            ), # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_370_',
        ), # (111, 128, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._06ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_371_',
    )  
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm17, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm16, # (111, 128, 16, 16)
                    name='relu_373_',
                ), # (111, 128, 16, 16)
                filters=tf.transpose(
                    self._06ParallelBlock_01SequentialBlock_07Conv2d_weight, # (512, 128, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_372_',
                ), # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_374_',
            ), # (111, 512, 16, 16)
            bias=self._06ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_375_',
        ), # (111, 512, 16, 16)
        scale=self._06ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._06ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_376_',
    )  
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._06ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val6 = tf.nn.relu(
        tf.add(
            batchnorm17, # (111, 512, 16, 16)
            val5, # (111, 512, 16, 16)
            name='add_377_',
        ), # (111, 512, 16, 16)
        name='relu_378_',
    )  
    (batchnorm18, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val6, # (111, 512, 16, 16)
                filters=tf.transpose(
                    self._07ParallelBlock_01SequentialBlock_01Conv2d_weight, # (128, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_379_',
                ), # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_380_',
            ), # (111, 128, 16, 16)
            bias=self._07ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_381_',
        ), # (111, 128, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._07ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_382_',
    )  
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm19, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm18, # (111, 128, 16, 16)
                name='relu_384_',
            ), # (111, 128, 16, 16)
            filters=tf.transpose(
                self._07ParallelBlock_01SequentialBlock_04Conv2d_weight, # (128, 128, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_383_',
            ), # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_385_',
        ), # (111, 128, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._07ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_386_',
    )  
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm20, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm19, # (111, 128, 16, 16)
                    name='relu_388_',
                ), # (111, 128, 16, 16)
                filters=tf.transpose(
                    self._07ParallelBlock_01SequentialBlock_07Conv2d_weight, # (512, 128, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_387_',
                ), # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_389_',
            ), # (111, 512, 16, 16)
            bias=self._07ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_390_',
        ), # (111, 512, 16, 16)
        scale=self._07ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._07ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_391_',
    )  
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._07ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val7 = tf.nn.relu(
        tf.add(
            batchnorm20, # (111, 512, 16, 16)
            val6, # (111, 512, 16, 16)
            name='add_392_',
        ), # (111, 512, 16, 16)
        name='relu_393_',
    )  
    (batchnorm21, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val7, # (111, 512, 16, 16)
                filters=tf.transpose(
                    self._08ParallelBlock_01SequentialBlock_01Conv2d_weight, # (128, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_394_',
                ), # (1, 1, 512, 128)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_395_',
            ), # (111, 128, 16, 16)
            bias=self._08ParallelBlock_01SequentialBlock_01Conv2d_bias, # (128)
            data_format='NCHW',
            name='bias_add_396_',
        ), # (111, 128, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (128)
        offset=self._08ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (128)
        mean=self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (128)
        variance=self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (128)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_397_',
    )  
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm22, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm21, # (111, 128, 16, 16)
                name='relu_399_',
            ), # (111, 128, 16, 16)
            filters=tf.transpose(
                self._08ParallelBlock_01SequentialBlock_04Conv2d_weight, # (128, 128, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_398_',
            ), # (3, 3, 128, 128)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_400_',
        ), # (111, 128, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (128)
        offset=self._08ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (128)
        mean=self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (128)
        variance=self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (128)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_401_',
    )  
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm23, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm22, # (111, 128, 16, 16)
                    name='relu_403_',
                ), # (111, 128, 16, 16)
                filters=tf.transpose(
                    self._08ParallelBlock_01SequentialBlock_07Conv2d_weight, # (512, 128, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_402_',
                ), # (1, 1, 128, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_404_',
            ), # (111, 512, 16, 16)
            bias=self._08ParallelBlock_01SequentialBlock_07Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_405_',
        ), # (111, 512, 16, 16)
        scale=self._08ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (512)
        offset=self._08ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (512)
        mean=self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (512)
        variance=self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_406_',
    )  
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._08ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val8 = tf.nn.relu(
        tf.add(
            batchnorm23, # (111, 512, 16, 16)
            val7, # (111, 512, 16, 16)
            name='add_407_',
        ), # (111, 512, 16, 16)
        name='relu_408_',
    )  
    (batchnorm24, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val8, # (111, 512, 16, 16)
                filters=tf.transpose(
                    self._09ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_409_',
                ), # (1, 1, 512, 256)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_410_',
            ), # (111, 256, 8, 8)
            bias=self._09ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_411_',
        ), # (111, 256, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._09ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_412_',
    )  
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm25, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm24, # (111, 256, 8, 8)
                name='relu_414_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._09ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_413_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_415_',
        ), # (111, 256, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._09ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_416_',
    )  
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm26, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm25, # (111, 256, 8, 8)
                    name='relu_418_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._09ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_417_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_419_',
            ), # (111, 1024, 8, 8)
            bias=self._09ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_420_',
        ), # (111, 1024, 8, 8)
        scale=self._09ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._09ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_421_',
    )  
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm27, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val8, # (111, 512, 16, 16)
            filters=tf.transpose(
                self._09ParallelBlock_02SequentialBlock_01Conv2d_weight, # (1024, 512, 1, 1)
                perm=[2, 3, 1, 0],
                name='transpose_422_',
            ), # (1, 1, 512, 1024)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_423_',
        ), # (111, 1024, 8, 8)
        scale=self._09ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (1024)
        offset=self._09ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (1024)
        mean=self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (1024)
        variance=self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_424_',
    )  
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._09ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val9 = tf.nn.relu(
        tf.add(
            batchnorm26, # (111, 1024, 8, 8)
            batchnorm27, # (111, 1024, 8, 8)
            name='add_425_',
        ), # (111, 1024, 8, 8)
        name='relu_426_',
    )  
    (batchnorm28, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val9, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._10ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_427_',
                ), # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_428_',
            ), # (111, 256, 8, 8)
            bias=self._10ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_429_',
        ), # (111, 256, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._10ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_430_',
    )  
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm29, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm28, # (111, 256, 8, 8)
                name='relu_432_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._10ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_431_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_433_',
        ), # (111, 256, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._10ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_434_',
    )  
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm30, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm29, # (111, 256, 8, 8)
                    name='relu_436_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._10ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_435_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_437_',
            ), # (111, 1024, 8, 8)
            bias=self._10ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_438_',
        ), # (111, 1024, 8, 8)
        scale=self._10ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._10ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_439_',
    )  
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._10ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val10 = tf.nn.relu(
        tf.add(
            batchnorm30, # (111, 1024, 8, 8)
            val9, # (111, 1024, 8, 8)
            name='add_440_',
        ), # (111, 1024, 8, 8)
        name='relu_441_',
    )  
    (batchnorm31, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val10, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._11ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_442_',
                ), # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_443_',
            ), # (111, 256, 8, 8)
            bias=self._11ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_444_',
        ), # (111, 256, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._11ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_445_',
    )  
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm32, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm31, # (111, 256, 8, 8)
                name='relu_447_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._11ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_446_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_448_',
        ), # (111, 256, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._11ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_449_',
    )  
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm33, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm32, # (111, 256, 8, 8)
                    name='relu_451_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._11ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_450_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_452_',
            ), # (111, 1024, 8, 8)
            bias=self._11ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_453_',
        ), # (111, 1024, 8, 8)
        scale=self._11ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._11ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_454_',
    )  
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._11ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val11 = tf.nn.relu(
        tf.add(
            batchnorm33, # (111, 1024, 8, 8)
            val10, # (111, 1024, 8, 8)
            name='add_455_',
        ), # (111, 1024, 8, 8)
        name='relu_456_',
    )  
    (batchnorm34, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val11, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._12ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_457_',
                ), # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_458_',
            ), # (111, 256, 8, 8)
            bias=self._12ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_459_',
        ), # (111, 256, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._12ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_460_',
    )  
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm35, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm34, # (111, 256, 8, 8)
                name='relu_462_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._12ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_461_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_463_',
        ), # (111, 256, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._12ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_464_',
    )  
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm36, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm35, # (111, 256, 8, 8)
                    name='relu_466_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._12ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_465_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_467_',
            ), # (111, 1024, 8, 8)
            bias=self._12ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_468_',
        ), # (111, 1024, 8, 8)
        scale=self._12ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._12ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_469_',
    )  
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._12ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val12 = tf.nn.relu(
        tf.add(
            batchnorm36, # (111, 1024, 8, 8)
            val11, # (111, 1024, 8, 8)
            name='add_470_',
        ), # (111, 1024, 8, 8)
        name='relu_471_',
    )  
    (batchnorm37, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val12, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._13ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_472_',
                ), # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_473_',
            ), # (111, 256, 8, 8)
            bias=self._13ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_474_',
        ), # (111, 256, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._13ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_475_',
    )  
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm38, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm37, # (111, 256, 8, 8)
                name='relu_477_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._13ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_476_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_478_',
        ), # (111, 256, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._13ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_479_',
    )  
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm39, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm38, # (111, 256, 8, 8)
                    name='relu_481_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._13ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_480_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_482_',
            ), # (111, 1024, 8, 8)
            bias=self._13ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_483_',
        ), # (111, 1024, 8, 8)
        scale=self._13ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._13ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_484_',
    )  
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._13ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val13 = tf.nn.relu(
        tf.add(
            batchnorm39, # (111, 1024, 8, 8)
            val12, # (111, 1024, 8, 8)
            name='add_485_',
        ), # (111, 1024, 8, 8)
        name='relu_486_',
    )  
    (batchnorm40, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val13, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._14ParallelBlock_01SequentialBlock_01Conv2d_weight, # (256, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_487_',
                ), # (1, 1, 1024, 256)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_488_',
            ), # (111, 256, 8, 8)
            bias=self._14ParallelBlock_01SequentialBlock_01Conv2d_bias, # (256)
            data_format='NCHW',
            name='bias_add_489_',
        ), # (111, 256, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (256)
        offset=self._14ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (256)
        mean=self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (256)
        variance=self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (256)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_490_',
    )  
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm41, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm40, # (111, 256, 8, 8)
                name='relu_492_',
            ), # (111, 256, 8, 8)
            filters=tf.transpose(
                self._14ParallelBlock_01SequentialBlock_04Conv2d_weight, # (256, 256, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_491_',
            ), # (3, 3, 256, 256)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_493_',
        ), # (111, 256, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (256)
        offset=self._14ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (256)
        mean=self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (256)
        variance=self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (256)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_494_',
    )  
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm42, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm41, # (111, 256, 8, 8)
                    name='relu_496_',
                ), # (111, 256, 8, 8)
                filters=tf.transpose(
                    self._14ParallelBlock_01SequentialBlock_07Conv2d_weight, # (1024, 256, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_495_',
                ), # (1, 1, 256, 1024)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_497_',
            ), # (111, 1024, 8, 8)
            bias=self._14ParallelBlock_01SequentialBlock_07Conv2d_bias, # (1024)
            data_format='NCHW',
            name='bias_add_498_',
        ), # (111, 1024, 8, 8)
        scale=self._14ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (1024)
        offset=self._14ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (1024)
        mean=self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (1024)
        variance=self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (1024)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_499_',
    )  
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._14ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val14 = tf.nn.relu(
        tf.add(
            batchnorm42, # (111, 1024, 8, 8)
            val13, # (111, 1024, 8, 8)
            name='add_500_',
        ), # (111, 1024, 8, 8)
        name='relu_501_',
    )  
    (batchnorm43, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val14, # (111, 1024, 8, 8)
                filters=tf.transpose(
                    self._15ParallelBlock_01SequentialBlock_01Conv2d_weight, # (512, 1024, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_502_',
                ), # (1, 1, 1024, 512)
                strides=[2, 2],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_503_',
            ), # (111, 512, 4, 4)
            bias=self._15ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_504_',
        ), # (111, 512, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._15ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_505_',
    )  
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm44, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm43, # (111, 512, 4, 4)
                name='relu_507_',
            ), # (111, 512, 4, 4)
            filters=tf.transpose(
                self._15ParallelBlock_01SequentialBlock_04Conv2d_weight, # (512, 512, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_506_',
            ), # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_508_',
        ), # (111, 512, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._15ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_509_',
    )  
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm45, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm44, # (111, 512, 4, 4)
                    name='relu_511_',
                ), # (111, 512, 4, 4)
                filters=tf.transpose(
                    self._15ParallelBlock_01SequentialBlock_07Conv2d_weight, # (2048, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_510_',
                ), # (1, 1, 512, 2048)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_512_',
            ), # (111, 2048, 4, 4)
            bias=self._15ParallelBlock_01SequentialBlock_07Conv2d_bias, # (2048)
            data_format='NCHW',
            name='bias_add_513_',
        ), # (111, 2048, 4, 4)
        scale=self._15ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (2048)
        offset=self._15ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (2048)
        mean=self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (2048)
        variance=self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_514_',
    )  
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    (batchnorm46, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            val14, # (111, 1024, 8, 8)
            filters=tf.transpose(
                self._15ParallelBlock_02SequentialBlock_01Conv2d_weight, # (2048, 1024, 1, 1)
                perm=[2, 3, 1, 0],
                name='transpose_515_',
            ), # (1, 1, 1024, 2048)
            strides=[2, 2],
            padding='VALID',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_516_',
        ), # (111, 2048, 4, 4)
        scale=self._15ParallelBlock_02SequentialBlock_02BatchNorm_gamma, # (2048)
        offset=self._15ParallelBlock_02SequentialBlock_02BatchNorm_beta, # (2048)
        mean=self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean, # (2048)
        variance=self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_517_',
    )  
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._15ParallelBlock_02SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    val15 = tf.nn.relu(
        tf.add(
            batchnorm45, # (111, 2048, 4, 4)
            batchnorm46, # (111, 2048, 4, 4)
            name='add_518_',
        ), # (111, 2048, 4, 4)
        name='relu_519_',
    )  
    (batchnorm47, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val15, # (111, 2048, 4, 4)
                filters=tf.transpose(
                    self._16ParallelBlock_01SequentialBlock_01Conv2d_weight, # (512, 2048, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_520_',
                ), # (1, 1, 2048, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_521_',
            ), # (111, 512, 4, 4)
            bias=self._16ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_522_',
        ), # (111, 512, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._16ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_523_',
    )  
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm48, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm47, # (111, 512, 4, 4)
                name='relu_525_',
            ), # (111, 512, 4, 4)
            filters=tf.transpose(
                self._16ParallelBlock_01SequentialBlock_04Conv2d_weight, # (512, 512, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_524_',
            ), # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_526_',
        ), # (111, 512, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._16ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_527_',
    )  
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm49, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm48, # (111, 512, 4, 4)
                    name='relu_529_',
                ), # (111, 512, 4, 4)
                filters=tf.transpose(
                    self._16ParallelBlock_01SequentialBlock_07Conv2d_weight, # (2048, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_528_',
                ), # (1, 1, 512, 2048)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_530_',
            ), # (111, 2048, 4, 4)
            bias=self._16ParallelBlock_01SequentialBlock_07Conv2d_bias, # (2048)
            data_format='NCHW',
            name='bias_add_531_',
        ), # (111, 2048, 4, 4)
        scale=self._16ParallelBlock_01SequentialBlock_08BatchNorm_gamma, # (2048)
        offset=self._16ParallelBlock_01SequentialBlock_08BatchNorm_beta, # (2048)
        mean=self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean, # (2048)
        variance=self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar, # (2048)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_532_',
    )  
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningMean.assign(running_mean)  
    self._16ParallelBlock_01SequentialBlock_08BatchNorm_runningVar.assign(running_var)  
    val16 = tf.nn.relu(
        tf.add(
            batchnorm49, # (111, 2048, 4, 4)
            val15, # (111, 2048, 4, 4)
            name='add_533_',
        ), # (111, 2048, 4, 4)
        name='relu_534_',
    )  
    (batchnorm50, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                val16, # (111, 2048, 4, 4)
                filters=tf.transpose(
                    self._17ParallelBlock_01SequentialBlock_01Conv2d_weight, # (512, 2048, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_535_',
                ), # (1, 1, 2048, 512)
                strides=[1, 1],
                padding='VALID',
                dilations=[1, 1],
                data_format='NCHW',
                name='convolution_536_',
            ), # (111, 512, 4, 4)
            bias=self._17ParallelBlock_01SequentialBlock_01Conv2d_bias, # (512)
            data_format='NCHW',
            name='bias_add_537_',
        ), # (111, 512, 4, 4)
        scale=self._17ParallelBlock_01SequentialBlock_02BatchNorm_gamma, # (512)
        offset=self._17ParallelBlock_01SequentialBlock_02BatchNorm_beta, # (512)
        mean=self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean, # (512)
        variance=self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar, # (512)
        epsilon=1.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_538_',
    )  
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningMean.assign(running_mean)  
    self._17ParallelBlock_01SequentialBlock_02BatchNorm_runningVar.assign(running_var)  
    (batchnorm51, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.convolution(
            tf.nn.relu(
                batchnorm50, # (111, 512, 4, 4)
                name='relu_540_',
            ), # (111, 512, 4, 4)
            filters=tf.transpose(
                self._17ParallelBlock_01SequentialBlock_04Conv2d_weight, # (512, 512, 3, 3)
                perm=[2, 3, 1, 0],
                name='transpose_539_',
            ), # (3, 3, 512, 512)
            strides=[1, 1],
            padding='SAME',
            dilations=[1, 1],
            data_format='NCHW',
            name='convolution_541_',
        ), # (111, 512, 4, 4)
        scale=self._17ParallelBlock_01SequentialBlock_05BatchNorm_gamma, # (512)
        offset=self._17ParallelBlock_01SequentialBlock_05BatchNorm_beta, # (512)
        mean=self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean, # (512)
        variance=self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar, # (512)
        epsilon=2.0E-5,
        is_training=True,
        exponential_avg_factor=0.9,
        data_format='NCHW',
        name='fused_batch_norm_542_',
    )  
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningMean.assign(running_mean)  
    self._17ParallelBlock_01SequentialBlock_05BatchNorm_runningVar.assign(running_var)  
    (batchnorm52, running_mean, running_var) = tf.compat.v1.nn.fused_batch_norm(
        tf.nn.bias_add(
            tf.nn.convolution(
                tf.nn.relu(
                    batchnorm51, # (111, 512, 4, 4)
                    name='relu_544_',
                ), # (111, 512, 4, 4)
                filters=tf.transpose(
                    self._17ParallelBlock_01SequentialBlock_07Conv2d_weight, # (2048, 512, 1, 1)
                    perm=[2, 3, 1, 0],
                    name='transpose_543_',
                ), # (1, 1, 512, 2048)
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
    )  
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
    )
    return result

## 1
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
    )
    return result

# number of epochs was 1
# number of prediction functions is 1
# number of loss functions is 1

