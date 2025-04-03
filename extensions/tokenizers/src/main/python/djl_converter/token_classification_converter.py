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

from djl_converter.huggingface_converter import HuggingfaceConverter


class TokenClassificationConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "token-classification"
        self.application = "nlp/token_classification"
        self.translator = "ai.djl.huggingface.translator.TokenClassificationTranslatorFactory"
        self.inputs = "My name is Wolfgang and I live in Berlin"
        self.outputs = ["wolfgang", "PER", "PROPN"]

    def verify_jit_output(self, hf_pipeline, encoding, out):
        config = hf_pipeline.model.config
        tokenizer = hf_pipeline.tokenizer

        logits = out["logits"][0].detach()
        input_ids = encoding["input_ids"][0].tolist()
        offset_mapping = encoding.encodings[0].offsets
        special_token_masks = encoding.encodings[0].special_tokens_mask
        probabilities = torch.softmax(logits, dim=1)
        entities = []

        for idx, scores in enumerate(probabilities):
            if special_token_masks[idx]:
                continue

            entity_idx = scores.argmax().item()
            entity = config.id2label[entity_idx]

            if entity != "O":
                item = {
                    "entity": entity,
                    "score": scores[entity_idx],
                    "index": idx,
                    "word": tokenizer.convert_ids_to_tokens(input_ids[idx]),
                    "start": offset_mapping[idx][0],
                    "end": offset_mapping[idx][1],
                }
                entities.append(item)

                if self.outputs[0] in item["word"].lower() and (
                        self.outputs[1] in entity
                        or self.outputs[2] in entity):
                    return True, None

        pipeline_output = hf_pipeline(self.inputs)
        if len(pipeline_output) == 0:
            logging.warning(f"Warning: pipeline output is empty")
            return True, None

        if len(entities) == 0:
            return False, "TokenClassification returns with empty result"

        for e in pipeline_output:
            if e["word"] == entities[0]["word"]:
                if e["entity"] == entities[0]["entity"]:
                    logging.warning(
                        f"pipeline output differs from expected: {pipeline_output}"
                    )
                    return True, None
                else:
                    break

        logging.error(f"Unexpected inference result: {entities[0]}")

        return False, "Unexpected inference result"
