#!/usr/bin/env python

# pip install git+https://github.com/ultralytics/CLIP.git
# pip install -U ultralytics

import json
import os
import urllib.request

import clip
import cv2
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from torch import nn
from ultralytics import YOLOWorld
from ultralytics.nn.modules import C2fAttn, WorldDetect, ImagePoolingAttn


class ClipModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.clip_model = model

    def forward(
            self,
            input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        txt_feats = self.clip_model.encode_text(input_tokens)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = txt_feats.unsqueeze(0)
        return text_feats


class VisionModelWrapper(nn.Module):

    def __init__(self, model, save) -> None:
        super().__init__()
        self.vision_model = model
        self.save = save

    def forward(
            self,
            txt_feats: torch.Tensor,
            x: torch.Tensor,
    ) -> torch.Tensor:
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.vision_model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        return x[0]


def save_serving_properties(model_name: str):
    serving_file = os.path.join(model_name, "serving.properties")
    arguments = {
        "engine":
            "PyTorch",
        "option.modelName":
            model_name,
        "option.mapLocation":
            "true",
        "toTensor":
            "true",
        "applyRatio":
            "true",
        "translatorFactory":
            "ai.djl.modality.cv.translator.YoloWorldTranslatorFactory",
    }

    with open(serving_file, 'w') as f:
        for k, v in arguments.items():
            f.write(f"{k}={v}\n")


def save_tokenizer(model_name: str):
    tokenizer = SimpleTokenizer()
    with open(f"{model_name}/vocab.json", "w", encoding='utf-8') as f:
        json.dump(tokenizer.encoder, f)

    sorted_dict = dict(sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1]))
    with open(f"{model_name}/merges.txt", "w", encoding='utf-8') as f:
        f.write("#version: 0.2")
        for key in sorted_dict.keys():
            f.write(f"\n{key[0]} {key[1]}")


def jit_trace():
    model_name = "yolov8s-worldv2"
    os.makedirs(model_name, exist_ok=True)
    save_tokenizer(model_name)

    classes = ["cat", "remote control"]
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(classes)
    model.predict("cat.jpg")  # this is necessary to setup model.predictor

    image = cv2.imdecode(np.fromfile("cat.jpg", np.uint8), cv2.IMREAD_COLOR)
    im = model.predictor.preprocess([image])
    text_tokens = clip.tokenize(classes)

    clip_wrapper = ClipModelWrapper(model.model.clip_model)
    vision_wrapper = VisionModelWrapper(model.model.model, model.model.save)
    txt_feats = clip_wrapper(text_tokens)
    preds = vision_wrapper(txt_feats, im)
    results = model.predictor.postprocess(preds, im, [image])
    print(results[0].names)

    traced_model = torch.jit.trace(clip_wrapper, text_tokens)
    torch.jit.save(traced_model, f"{model_name}/clip.pt")

    traced_model = torch.jit.trace(vision_wrapper, [txt_feats, im])
    torch.jit.save(traced_model, f"{model_name}/{model_name}.pt")

    save_serving_properties(model_name)
    print("trace yolov8s-worldv2 model finished.")


if __name__ == "__main__":
    if not os.path.exists("cat.jpg"):
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        urllib.request.urlretrieve(image_url, "cat.jpg")

    jit_trace()
