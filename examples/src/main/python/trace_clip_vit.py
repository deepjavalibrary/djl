#!/usr/bin/env python

import os
from collections import OrderedDict
from typing import Any

import requests
import torch
from PIL import Image
from torch import nn
from transformers import pipeline


class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> dict[Any, torch.Tensor]:
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values)
        # filter non-Tensor
        ret = OrderedDict()
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v
        return ret


def jit_trace():
    model_id = "openai/clip-vit-large-patch14"
    pipe = pipeline(model=model_id, framework="pt", device="cpu")

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    texts = ["cat", "remote control"]
    hypothesis_template = "This is a photo of {}."
    sequences = [hypothesis_template.format(x) for x in texts]

    encoding = pipe.tokenizer(sequences, padding=True, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    image_features = pipe.image_processor(images=image, return_tensors='pt')
    pixel_values = image_features["pixel_values"]

    traced_model = torch.jit.trace(ModelWrapper(pipe.model),
                                   (input_ids, attention_mask, pixel_values),
                                   strict=False)

    model_name = model_id.split("/")[1]
    os.makedirs(model_name, exist_ok=True)

    pipe.tokenizer.save_pretrained(model_name)
    for path in os.listdir(model_name):
        if path != "tokenizer.json" and path != "tokenizer_config.json":
            os.remove(os.path.join(model_name, path))

    torch.jit.save(traced_model, f"{model_name}/{model_name}.pt")
    serving_file = os.path.join(model_name, "serving.properties")
    arguments = {
        "engine":
        "PyTorch",
        "option.modelName":
        model_name,
        "option.mapLocation":
        "true",
        "width":
        "224",
        "height":
        "224",
        "resizeShort":
        "true",
        "toTensor":
        "true",
        "centerCrop":
        "true",
        "normalize":
        "0.48145466,0.4578275,0.40821073,0.26862954,0.26130258,0.27577711",
        "translatorFactory":
        "ai.djl.huggingface.translator.ZeroShotImageClassificationTranslatorFactory",
    }

    with open(serving_file, 'w') as f:
        for k, v in arguments.items():
            f.write(f"{k}={v}\n")


if __name__ == '__main__':
    # main()
    jit_trace()
