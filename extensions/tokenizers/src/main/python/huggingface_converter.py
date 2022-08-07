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
import os.path
import sys
from arg_parser import converter_args
from metadata import HuggingfaceMetadata
from zip_utils import zip_dir
import shutil
from shasum import sha1_sum

import json
from huggingface_hub import HfApi, ModelSearchArguments
from huggingface_hub import hf_hub_download
from transformers import pipeline
import torch

SUPPORTED_TASK = {
    "fill-mask": {
        "translator":
        "ai.djl.huggingface.translator.FillMaskTranslatorFactory",
        "application":
        "nlp/fill_mask",
        "inputs":
        "Hello I'm a [MASK] model.",
        "output": [
            "fashion", "role", 'new', 'super', 'fine', 'male', 'female', 'big',
            'top', 'modeling', 'virtual'
        ]
    },
    "question-answering": {
        "translator":
        "ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory",
        "application": "nlp/question_answer",
        "inputs": {
            "question":
            "When did BBC Japan start broadcasting?",
            "context":
            "BBC Japan was a general entertainment Channel. Which operated between December 2004 and April 2006."
            " It ceased operations after its Japanese distributor folded. "
        },
        "output": "december 2004"
    },
    "text-classification": {
        "translator":
        "ai.djl.huggingface.translator.SentimentAnalysisTranslatorFactory",
        "application": "nlp/text_classification",
        "inputs": "I like DJL. DJL is the best DL framework!"
    },
    "token-classification": {
        "translator":
        "ai.djl.huggingface.translator.TokenClassificationTranslatorFactory",
        "application": "nlp/token_classification",
        "inputs": "I like DJL. DJL is the best DL framework!"
    },
}

ARCHITECTURES_2_TASK = {
    "ForQuestionAnswering": "question-answering",
    "ForTokenClassification": "token-classification",
    "ForSequenceClassification": "text-classification",
    "ForMultipleChoice": "text-classification",
    "ForMaskedLM": "fill-mask"
}


class HuggingfaceConverter:

    def __init__(self, limit: int, output_dir: str):
        self.limit = limit
        self.output_dir = output_dir
        self.processed_models = {}
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        output_path = os.path.join(output_dir, "processed_models.json")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                self.processed_models = json.load(f)

    def convert_huggingface_models(self, category: str):
        languages = ModelSearchArguments().language
        api = HfApi()
        models = api.list_models(filter=f"{category},pytorch",
                                 sort="downloads",
                                 direction=-1,
                                 limit=self.limit)

        temp_dir = f"{self.output_dir}/tmp"

        for model_info in models:
            model_id = model_info.modelId
            is_english = True
            for tag in model_info.tags:
                if tag in languages and tag != 'en':
                    is_english = False
                    break

            if not is_english:
                continue

            if self.processed_models.get(model_id):
                logging.info(f"Skip converted mode: {model_id}.")
                continue

            result, reason, size = self.save_model(model_id)
            if not result:
                logging.error(reason)

            self.save_progress(model_id, model_info.sha, result, reason, size)
            shutil.rmtree(temp_dir)

    def save_model(self, model_id: str):
        config = hf_hub_download(repo_id=model_id, filename="config.json")
        task, architecture = self.to_supported_task(config)
        if not task:
            return False, "Unsupported model architecture: {architecture}", -1

        temp_dir = f"{self.output_dir}/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        hf_pipeline = self.load_model(task, model_id)
        # Save tokenizer.json
        self.save_tokenizer(hf_pipeline, temp_dir)
        # Save config.json just for reference
        shutil.copyfile(config, os.path.join(temp_dir, "config.json"))
        # Save jit traced .pt file
        model_file = self.jit_trace_model(hf_pipeline, task, model_id,
                                          temp_dir)
        if not model_file:
            return False, "Failed to trace model", -1

        result, reason = self.verify_jit_model(hf_pipeline, task, model_file)
        if not result:
            return False, reason, -1

        size = self.save_model_zoo(task, model_id, temp_dir,
                                   hf_pipeline.tokenizer.mask_token)

        return True, None, size

    @staticmethod
    def save_tokenizer(hf_pipeline, temp_dir: str):
        hf_pipeline.tokenizer.save_pretrained(temp_dir)
        # only keep tokenizer.json file
        for path in os.listdir(temp_dir):
            if path != "tokenizer.json":
                os.remove(os.path.join(temp_dir, path))

    def jit_trace_model(self, hf_pipeline, task: str, model_id: str,
                        temp_dir: str):
        input_ids, attention_mask = self.encode_inputs(task,
                                                       hf_pipeline.tokenizer)

        # noinspection PyBroadException
        try:
            script_module = torch.jit.trace(hf_pipeline.model,
                                            (input_ids, attention_mask),
                                            strict=False)
        except:
            try:
                script_module = torch.jit.trace(hf_pipeline.model,
                                                input_ids,
                                                strict=False)
            except Exception as e:
                logging.warning(f"Failed to trace model: {model_id}.")
                logging.warning(e, exc_info=True)
                return None

        model_name = model_id.split("/")[-1]
        model_file = os.path.join(temp_dir, f"{model_name}.pt")
        script_module.save(model_file)
        return model_file

    def save_model_zoo(self, task: str, model_id: str, temp_dir: str,
                       mask_token: str):
        artifact_ids = model_id.split("/")
        model_name = artifact_ids[-1]

        application = SUPPORTED_TASK[task]["application"]
        translator = SUPPORTED_TASK[task]["translator"]
        repo_dir = f"{self.output_dir}/model/{application}/ai/djl/huggingface/{model_id}"
        model_dir = f"{repo_dir}/0.0.1"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save serving.properties
        serving_file = os.path.join(temp_dir, "serving.properties")
        with open(serving_file, 'w') as f:
            f.write(f"engine=PyTorch\n"
                    f"option.modelName={model_name}\n"
                    f"option.mapLocation=true\n"
                    f"maskToken={mask_token}\n"
                    f"translatorFactory={translator}")

        # Save model as .zip file
        zip_file = os.path.join(model_dir, f"{model_name}.zip")
        zip_dir(temp_dir, zip_file)

        # Save metadata.json

        sha1 = sha1_sum(zip_file)
        file_size = os.path.getsize(zip_file)
        metadata = HuggingfaceMetadata(artifact_ids, application, translator,
                                       sha1, file_size)
        metadata_file = os.path.join(repo_dir, "metadata.json")
        metadata.save_metadata(metadata_file)

        return file_size

    def verify_jit_model(self, hf_pipeline, task: str, model_file: str):
        expected_output = SUPPORTED_TASK[task]["output"]
        input_ids, attention_mask = self.encode_inputs(task,
                                                       hf_pipeline.tokenizer)

        if torch.cuda.is_available():
            traced_model = torch.jit.load(model_file, map_location='cuda:0')
            traced_model.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        else:
            traced_model = torch.jit.load(model_file)

        traced_model.eval()

        # test traced model
        out = traced_model(input_ids, attention_mask)
        tokenizer = hf_pipeline.tokenizer

        if task == "question-answering":
            start_ = out["start_logits"]
            end_ = out["end_logits"]
            start_[0, 0] = 0
            end_[0, 0] = 0
            answer_start = torch.argmax(start_)
            answer_end = torch.argmax(end_) + 1

            out_ids = input_ids[0].tolist()[answer_start:answer_end]
            tokens = tokenizer.convert_ids_to_tokens(out_ids)
            prediction = tokenizer.convert_tokens_to_string(tokens).strip()

            if prediction.lower() != expected_output:
                inputs = SUPPORTED_TASK[task]["inputs"]
                pipeline_output = hf_pipeline(inputs)
                if pipeline_output != prediction:
                    return False, f"Unexpected inference result: {prediction}"
                else:
                    logging.warning(
                        f"pipeline output differs from expected: {pipeline_output}"
                    )
        elif task == "fill-mask":
            mask_token_id = tokenizer.mask_token_id
            mask_index = torch.nonzero(input_ids.squeeze(0) == mask_token_id,
                                       as_tuple=False).squeeze(0)
            logits = out['logits'][0, mask_index]
            answer = torch.argmax(logits)
            prediction = tokenizer.decode(answer).strip()

            if prediction not in expected_output:
                inputs = SUPPORTED_TASK[task]["inputs"]
                if tokenizer.mask_token != "[MASK]":
                    inputs = inputs.replace("[MASK]", tokenizer.mask_token)
                pipeline_output = hf_pipeline(inputs)

                if prediction not in [o["token_str"] for o in pipeline_output]:
                    return False, f"Unexpected inference result: {prediction}"
                else:
                    logging.warning(
                        f"pipeline output differs from expected: {pipeline_output}"
                    )
        else:
            if not hasattr(out, "last_hidden_layer"):
                return False, f"Unexpected inference result: {out}"

        return True, None

    @staticmethod
    def load_model(task: str, model_id: str):
        logging.info(f"Loading model: {model_id}.")
        kwargs = {
            "tokenizer": model_id,
            "device": -1  # always use CPU to trace the model
        }
        hf_pipeline = pipeline(task=task,
                               model=model_id,
                               framework="pt",
                               **kwargs)
        return hf_pipeline

    @staticmethod
    def encode_inputs(task: str, tokenizer):
        inputs = SUPPORTED_TASK[task]["inputs"]
        if type(inputs) is dict:
            text = inputs["question"]
            text_pair = inputs["context"]
        else:
            text = inputs
            text_pair = None

        if task == "fill-mask" and tokenizer.mask_token != "[MASK]":
            text = text.replace("[MASK]", tokenizer.mask_token)

        encoding = tokenizer.encode_plus(text,
                                         text_pair=text_pair,
                                         return_tensors='pt')
        return encoding["input_ids"], encoding["attention_mask"]

    @staticmethod
    def to_supported_task(config: str):
        with open(config) as f:
            config = json.load(f)
            architecture = config.get("architectures", [None])[0]
            for arch in ARCHITECTURES_2_TASK:
                if architecture.endswith(arch):
                    return ARCHITECTURES_2_TASK[arch], architecture

            return None, architecture

    def save_progress(self, model_id: str, sha: str, result: bool, reason: str,
                      size: int):
        status = {
            "result": "success" if result else "failed",
            "sha1": sha,
            "size": size
        }
        if reason:
            status["reason"] = reason
        self.processed_models[model_id] = status

        dict_file = os.path.join(self.output_dir, "processed_models.json")
        with open(dict_file, 'w') as f:
            json.dump(self.processed_models,
                      f,
                      sort_keys=True,
                      indent=2,
                      ensure_ascii=False)


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    args = converter_args()
    converter = HuggingfaceConverter(args.limit, args.output_dir)
    for category in args.categories:
        converter.convert_huggingface_models(category)

    logging.info("finished.")


if __name__ == "__main__":
    main()
