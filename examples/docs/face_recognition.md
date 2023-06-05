# Face recognition example

In this example, you learn how to implement inference code with a pytorch model to extract and compare face features.

Extract face feature:
The source code can be found at [FeatureExtraction.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/face/FeatureExtraction.java).  
The model github can be found at [facenet-pytorch](https://github.com/timesler/facenet-pytorch).

Compare face features:
The source code can be found at [FeatureComparison.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/face/FeatureComparison.java).  

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run face recognition example

### Input image file
You can find the image used in this example in the project test resource folder:  
 `src/test/resources/kana1.jpg`  
![kana1](../src/test/resources/kana1.jpg)     
 `src/test/resources/kana2.jpg`  
![kana2](../src/test/resources/kana2.jpg)  

### Build the project and run
Use the following command to run the project:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.face.FeatureExtraction
```

Your output should look like the following:

```text
[INFO ] - [-0.04026184, -0.019486362, -0.09802659, 0.01700999, 0.037829027, ...]
```

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.face.FeatureComparison
```

Your output should look like the following:

```text
[INFO ] - 0.9022607
```

## Reference - how to import pytorch model:

1. Install:
    
```bash
# With pip:
pip install facenet-pytorch

# or clone this repo, removing the '-' to allow python imports:
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

```
    
2. In python, import facenet-pytorch and instantiate model, then use torch.jit.trace to generate a torch.jit.ScriptModule via tracing:
    
```python
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open('/path/to/any/face/image.jpg')

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 320, 320)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(resnet, example)

# For control flow, use script
#script_module = torch.jit.script(model) 

# Save the TorchScript model
traced_script_module.save("face_feature.pt")

output = traced_script_module(torch.rand(1,3,320, 320))
#print(traced_script_module.code)
print(output)

```
