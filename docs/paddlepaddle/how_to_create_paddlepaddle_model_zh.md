# 如何导入Paddle模型

PaddlePaddle的模型来源有很多种。你可以选择直接从 PaddleHub 下载 或者直接从现有代码中导出。
在这个教学中，我们会介绍两种主要的转换途径：

- PaddleHub: PaddlePaddle提供的模型库
- 来自代码的 Paddle 模型 

## PaddleHub

[PaddleHub](https://github.com/PaddlePaddle/PaddleHub) 是一个 PaddlePaddle提供的预训练模型库。里面包含了大量业内常用的深度学习模型。

要在DJL中使用PaddleHub模型只需要以下四步:

1. 从 PaddleHub [模型搜索](https://www.paddlepaddle.org.cn/hublist?filter=hot&value=1) 中寻找要的模型
2. 从 "代码示例" 里获取模型读取方法
3. 将代码复制到python环境中并导出
4. 在 DJL 进行推理任务

我们用一个基本的图片分类模型为例。假设我们想要一个训练好的 MobileNet 模型,
就可以从下面的链接找到:

[MobileNet model](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v3_large_imagenet_ssld&en_category=ImageClassification)

然后在 "代码示例" 找到代码

```
import paddlehub as hub
import cv2

classifier = hub.Module(name="mobilenet_v3_large_imagenet_ssld")

result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
```

接下来就是复制到你的python环境里 (确保已经安装 PaddleHub ).
上面的代码会下载模型并使用一张图片做推理测试。 如果想试验一下效果的话,
只需要替换 `'/PATH/TO/IMAGE'` 到你的本地图片路径。

接下来，我们只需要添加以下一行到之前的代码上:

```
module.save_inference_model(dirname="model/mobilenet")
```

它会将模型保存成可以部署的格式。 产生的文件结构如下图所示:

```
- model
  - mobilenet
    - __model__
    - __params__
```

`__model__` 和 `__params__` 是推理用的必要文件。接下来只需要将 `mobilenet` 打包成zip。

最后, 你可以直接使用 `mobilenet.zip` 在 DJL 中进行推理。 

总结, 以下两行就是在 PaddleHub 中转换模型的泛用模版:

```
import paddlehub as hub

model = hub.Module(name="modelname")
model.save_inference_model(dirname="model/modelname")
```

## Paddle 模型 (包含预训练权重)

如果你手里有代码以及Paddle的预训练权重, 或者有训练代码,
你可以使用下面的代码来完成模型转换。

首先 我们假设你拥有模型构建的代码, 同时模型的权重已经训练/加载完成.

### Paddle 动态图模型 (2.0)

Paddle 2.0 的动态图模型可用如下代码表达:

```
class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)

layer = LinearNet()

path = "model/inference"
paddle.jit.save(layer, path)
```

`paddle.jit.save` 可以将你的模型保存成:

```
- model
  - inference.pdmodel
  - inference.pdiparams
```
之后只需要将 `model` 文件夹打包就可以在DJL用了。 注意，你需要确保模型名字是
`inference.*`。DJL在模型读取时只会读取这个名字的模型。

### Paddle 静态图模型 (1.x)

对于 2.0 以前的Paddle模型, 它们会是静态图的格式:

```
import paddle

paddle.enable_static()

path_prefix = "./inference"

# User defined network, here a softmax regession example
image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
predict = paddle.static.nn.fc(image, 10, activation='softmax')

loss = paddle.nn.functional.cross_entropy(predict, label)

exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(paddle.static.default_startup_program())

# Feed data and train process

# Save inference model. Note we don't save label and loss in this example
paddle.static.save_inference_model(path_prefix, [image], [predict], exe)
```

可以使用 `paddle.static.save_inference_model` 来保存模型。
同样的, 最后只需要打包成zip就可以在 DJL 推理了。



## 使用 DJL 推理

首先假设你已经有了一个 Paddle 模型压缩包, 你可以直接用下面的代码载入模型并进行 NDArray 输入/输出的推理:

```java
// load model
Criteria<NDList, NDList> criteria = Criteria.builder()
        .setsetTypes(NDList.class, NDList.class)
        .optModelPath("your/path/to/zip")
        .optModelName("your folder name insize zip")
        .build();

ZooModel<NDList, NDList> model = criteria.loadModel();
// run inference
Predictor<NDList, NDList> predictor = model.newPredictor();
NDManager manager = NDManager.newBaseManager()
NDList list ; // the input for your models as pure NDArray
predictor.predict(list);
```

在这里，你需要知道模型的输入输出格式, 比如图片经常表达成 NCHW (批大小, RGB通道, 高度, 宽度)的多维矩阵。

虽然这样可以让模型跑起来, 但是最好还是结合 DJL 的 `Translator` class 使用。你可以在 [这里](../../jupyter/paddlepaddle/face_mask_detection_paddlepaddle.ipynb) 找到一些示例代码。
