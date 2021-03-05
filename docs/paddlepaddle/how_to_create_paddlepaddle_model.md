This document will show you how to train a model using the Paddle 2.0.

We refer to [LeNet's MNIST Dataset Image Classification](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/cv_case/image_classification/image_classification.html#lenetmnist), and use the Paddle 2.0 interface to train a simple The models are stored as pre-training and predictive deployment models. We will focus on how to generate model files. 

- Dependent package import

```
import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor
```

- View the version of Paddle

```
print(paddle.__version__)
```

- Dataset preparation

```
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
```

- Build a LeNet network

```
class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
```

- Model training

```
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
model = LeNet()
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
def train(model, optim):
    model.train()
    epochs = 2
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
train(model, optim)
```

-Store training model (training format): You can refer to [Parameter Storage](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/02_paddle2.0_develop/08_model_save_load_cn.html#id8) to learn how store the training format model under the dynamic graph. Just call the `paddle.save` interface.

```
paddle.save(model.state_dict(), 'lenet.pdparams')
paddle.save(optim.state_dict(), "lenet.pdopt")
```