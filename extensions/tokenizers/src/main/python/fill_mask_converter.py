#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging

import torch

from huggingface_converter import HuggingfaceConverter


class FillMaskConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "fill-mask"
        self.application = "nlp/fill_mask"
        self.translator = "ai.djl.huggingface.translator.FillMaskTranslatorFactory"
        self.inputs = "Hello I'm a [MASK] model."
        self.outputs = [
            "fashion", "role", 'new', 'super', 'fine', 'male', 'female', 'big',
            'top', 'modeling', 'virtual'
        ]

    def verify_jit_output(self, hf_pipeline, encoding, out):
        tokenizer = hf_pipeline.tokenizer
        mask_token_id = tokenizer.mask_token_id
        mask = encoding["input_ids"].squeeze(0) == mask_token_id

        mask_index = torch.nonzero(mask, as_tuple=False).squeeze(0)
        logits = out['logits'][0, mask_index]
        answer = torch.argmax(logits)
        prediction = tokenizer.decode(answer).strip()

        if prediction not in self.outputs:
            text = self.inputs
            if tokenizer.mask_token != "[MASK]":
                text = text.replace("[MASK]", tokenizer.mask_token)
            pipeline_output = hf_pipeline(text)

            if prediction not in [o["token_str"] for o in pipeline_output]:
                logging.error(f"Unexpected inference result: {prediction}")
                return False, "Unexpected inference result"

            logging.warning(
                f"pipeline output differs from expected: {pipeline_output}")

        return True, None

    def encode_inputs(self, tokenizer):
        text = self.inputs.replace("[MASK]", tokenizer.mask_token)
        return tokenizer.encode_plus(text, return_tensors='pt')

    def get_extra_arguments(self, hf_pipeline, model_id: str,
                            temp_dir: str) -> dict:
        return {"maskToken": hf_pipeline.tokenizer.mask_token}
