# CLIP model example

[CLIP model](https://huggingface.co/openai/clip-vit-base-patch32) is open-sourced by OpenAI for text-image understanding.
It is widely used to get text and image feature and used for search domain.  User could use image to search image, use text to search image and even use text to search text with this model.

In this short demo, we will do an image to text comparison to find which text is close to the corresponding image.

The image we used is

![](http://images.cocodataset.org/val2017/000000039769.jpg)

And our input text:

```
"A photo of cats";
"A photo of dogs";
```

We expect cats text will win based on the image.

## Run the example

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.clip.ImageTextComparison
```

output:

```
[INFO ] - A photo of cats Probability: 0.9970879546345841
[INFO ] - A photo of dogs Probability: 0.002912045365415886

```

## Trace the model

The current model can only run on CPU. To trace the model on specific gpu device, please follow the below instruction:

```python
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import requests
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name, torchscript=True, return_dict=False)
processor = CLIPProcessor.from_pretrained(model_name)

# put the model on specific gpu device:
# model.to('cuda:0')

test_text = "this is a cat"
text_inputs = processor(text=test_text, return_tensors="pt")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_inputs = processor(images=image, return_tensors="pt")

inputs = processor(text=["a photo of a cat", "a photo of dogs"], images=image, return_tensors="pt", padding=True)

converted = torch.jit.trace_module(model,  {'get_text_features': [text_inputs['input_ids'], text_inputs['attention_mask']],
'get_image_features': [image_inputs['pixel_values']],
'forward': [text_inputs['input_ids'], image_inputs['pixel_values'], text_inputs['attention_mask']]})

torch.jit.save(converted, "cliptext.pt")
```

The traced model can be run on the device user defined.
