# How to import PaddlePaddle model

There are many sources to use PaddlePaddle model in DJL. You can choose to pick from PaddleHub or directly export from the training logic.
In this section, we will go over two ways to load your Paddle model:

- PaddleHub: a model hub contains bunch of the pretrained model
- Paddle model from code

## PaddleHub

[PaddleHub](https://github.com/PaddlePaddle/PaddleHub) is a model hub provided by PaddlePaddle that includes many pretrained model
from different categories, from general CV/NLP model to speech analysis and even video classification models.

To use PaddleHub model in DJL, all you need to do is the following steps:

1. Find your model from the PaddleHub [Search Engine](https://www.paddlepaddle.org.cn/hublist?filter=hot&value=1)
2. Find "代码示例" section once you click on the model
3. Copy the code in python and do conversion
4. Do inference with DJL

We use an Image Classification as an example. Saying if we would like to use a pretrained MobileNet model for inference,
just go to the following link:

[MobileNet model](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v3_large_imagenet_ssld&en_category=ImageClassification)

Then we find "代码示例" section here:

```
import paddlehub as hub
import cv2

classifier = hub.Module(name="mobilenet_v3_large_imagenet_ssld")

result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
```

Then we just copy these piece of code in python (make sure PaddleHub is installed).
The above code will download the model and run inference test with an image. If you would like to test the model before using,
please replace the `'/PATH/TO/IMAGE'` to your local image path.

Then, all we need to do is appending one more line to the previous code:

```
module.save_inference_model(dirname="model/mobilenet")
```

This line will save the model into the Paddle format that allow Service deployment. You will find the following file being generated from your project:

```
- model
  - mobilenet
    - __model__
    - __params__
```

`__model__` and `__params__` are the key components for your inference. Then all you need to do is zip the `mobilenet` directory.

Finally, you can directly feed the `mobilenet.zip` file in DJL for inference task. 

As a summary, here is the pattern for you to save the model in the rest of PaddleHub:

```
import paddlehub as hub

model = hub.Module(name="modelname")
model.save_inference_model(dirname="model/modelname")
```

## Paddle model (with pretrained weight)

If you find some code with pretrained weight, or you just finished your training on PaddlePddle,
you can follow this instruction to save your model into Paddle Serving ready format.

Firstly let's assume you have code, and you already load the pretrained weight.

### Paddle Imperative model (2.0)

For imperative model trained using Paddle 2.0 like below:

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

`paddle.jit.save` will save the model like:

```
- model
  - inference.pdmodel
  - inference.pdiparams
```
Then just zip the model folder and your model is ready to use in DJL. However, you have to make sure the model prefix
is `inference.*` since DJL will only find files with this prefix.

### Paddle Symbolic model (1.x)

For Paddle model created before 2.0, it is usually in Symbolic form:

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

You can use `paddle.static.save_inference_model` to save the model.
Similarly, just zip the folder and run inference with DJL.



## Inference in DJL

Let's assume you have a zip file that contains Paddle model, you can just run inference with NDArray in/out mode like follows:

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

As mentioned, you need to find out what is the input for the model, like images usually interpret as NCHW (batch_size, channel, height, width).

However, usage like this is really basic, you can write a `Translator` in DJL for it. You can find some code examples [here](../../jupyter/paddlepaddle/face_mask_detection_paddlepaddle.ipynb).

