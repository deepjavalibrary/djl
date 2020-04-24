# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
import torch
import transformers
import argparse
import os

parser = argparse.ArgumentParser(description='This is the huggingface model export portal')
parser.add_argument("--name", required=True, type=str, help="This is the name of the available huggingface models")
parser.add_argument("--model_application", required=True, type=str, help="The application of huggingface models")
parser.add_argument("--shape", required=True, type=str, help="This is the input shape of the model, like [(1, 3, 224, 224), (1, 2, 3)]")
parser.add_argument("--output_path", required=True, type=str, help="This is the output path of the model")

def export(name, model_application, inputs, export_path):
    model = getattr(transformers, model_application).from_pretrained(name, torchscript=True)
    traced_model = torch.jit.trace(model, inputs)
    traced_model.eval()
    export_path = os.path.join(export_path, name)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    traced_model.save(os.path.join(export_path, name + '.pt'))

if __name__ == "__main__":
    args = parser.parse_args()
    tensors = []
    for shape in eval(args.shape):
        tensors.append(torch.ones(shape, dtype=torch.int64))
    export(args.name, args.model_application, tuple(tensors), args.output_path)
