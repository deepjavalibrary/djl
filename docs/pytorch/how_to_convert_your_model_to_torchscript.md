## How to convert your PyTorch model to TorchScript

There are two ways to convert your model to TorchScript: tracing and scripting.
We will only demonstrate the first one, tracing, but you can find information about scripting from the PyTorch documentation.
When tracing, we use an example input to record the actions taken and capture the the model architecture.
This works best when your model doesn't have control flow. 
If you do have control flow, you will need to use the scripting approach.
In DJL, we use tracing to create TorchScript for our ModelZoo models.


## Basic conversion

Here is an example of tracing in actions:

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")
```

Then you can just use the saved model in DJL like all other models.

## Advanced case

In some cases, you may have a method name that is not `forward` in pytorch (HuggingFace) like one below:

```python
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import requests
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name, torchscript=True, return_dict=False)
processor = CLIPProcessor.from_pretrained(model_name)

test_text = "this is a cat"
text_inputs = processor(text=test_text, return_tensors="pt")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_inputs = processor(images=image, return_tensors="pt")

converted = torch.jit.trace_module(model,  {'get_text_features': [text_inputs['input_ids'], text_inputs['attention_mask']],
                                            'get_image_features': [image_inputs['pixel_values']]})

torch.jit.save(converted, "cliptext.pt")
```

You can trace by using the `torch.traceModule` function. 

To run inference with such model in DJL, you could provide a placeholder NDArray like below:

```
NDArray array = NDManager.create("");
array.setName("module_method:get_text_features");
```

to tell which entryPoint method you would like to use. The placeholder NDArray will NOT be used in inference.

For more information on exporting TorchScript, see the [Loading a torch script model in C++ tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html).
