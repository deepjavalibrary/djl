## How to convert your Gluon model to an MXNet Symbol

DJL currently supports symbolic model loading from MXNet.
A gluon [HybridBlock](https://mxnet.apache.org/api/python/docs/api/gluon/hybrid_block.html) can be converted into a symbol for loading by doing as follows:

```python
from mxnet import nd
from mxnet.gluon import nn

# create a simple HybridSequential block
net = nn.HybridSequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))

# initialize and hybridize the block
net.initialize()
net.hybridize(static_alloc=True, static_shape=True)

# create sample input and run forward once
x = nd.random.uniform(shape=(2, 20))
net(x)

# export your model
net.export("sample_model")
```

After this is run, you will find `sample_model-0000.params` and `sample_model-symbol.json` in your local path.
These can be loaded in DJL.

In real applications, you may want to create and train a HybridBlock before exporting it.
The code block below shows how you can convert a [GluonCV](https://gluon-cv.mxnet.io/) pretrained model:

```python
import mxnet as mx
from gluoncv import model_zoo

# get the pretrained model from the gluon model zoo
net = model_zoo.get_model('resnet18_v1', pretrained=True)
net.hybridize(static_alloc=True, static_shape=True)

# create a sample input and run forward once (required for tracing)
x = nd.random.uniform(shape=(1, 3, 224, 224))
net(x)

# export your model
net.export("sample_model")
```

### hybridize without `static_alloc=True, static_shape=True`

It is always recommended enabling the static settings when exporting Apache MXNet model. This will ensure DJL to have the best performance for inference.

If you run hybridize without `static_alloc=True, static_shape=True`:

```python
net.hybridize()
```

you can enable this Java property with DJL:

```
-Dai.djl.mxnet.static_alloc=False -Dai.djl.mxnet.static_shape=False
```

This will ensure we skip the static settings in the inference model and make DJL produce consistent result with Python.

