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
import json
import logging
import os
from argparse import Namespace
from typing import List

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub.hf_api import ModelInfo
from djl_converter.fill_mask_converter import FillMaskConverter
from djl_converter.metadata import get_lang_tags
from djl_converter.question_answering_converter import QuestionAnsweringConverter
from djl_converter.sentence_similarity_converter import SentenceSimilarityConverter
from djl_converter.text_classification_converter import TextClassificationConverter
from djl_converter.token_classification_converter import TokenClassificationConverter

ARCHITECTURES_2_TASK = {
    "ForQuestionAnswering": "question-answering",
    "ForTokenClassification": "token-classification",
    "ForSequenceClassification": "text-classification",
    "ForMultipleChoice": "text-classification",
    "ForMaskedLM": "fill-mask",
}
SUPPORTED_TASKS = {
    "fill-mask": FillMaskConverter(),
    "question-answering": QuestionAnsweringConverter(),
    "sentence-similarity": SentenceSimilarityConverter(),
    "text-classification": TextClassificationConverter(),
    "token-classification": TokenClassificationConverter(),
}


class HuggingfaceModels:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.processed_models = {}

        output_path = os.path.join(output_dir, "models.json")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                self.processed_models = json.load(f)

        self.temp_dir = f"{self.output_dir}/tmp"

    def list_models(self, args: Namespace) -> List[dict]:
        import_all = os.environ.get("HF_IMPORT_ALL")

        api = HfApi()
        if args.model_name:
            all_models = api.list_models(search=args.model_name,
                                         sort="downloads",
                                         direction=-1,
                                         limit=args.limit)
            import_all = True
        else:
            all_models = api.list_models(filter=args.category,
                                         sort="downloads",
                                         direction=-1,
                                         limit=args.limit)
        models = [
            model for model in all_models
            if 'pytorch' in model.tags or 'safetensors' in model.tags
        ]
        if not models:
            if args.model_name:
                logging.warning(f"no model found: {args.model_name}.")
            else:
                logging.warning(f"no model matches category: {args.category}.")

            return []

        ret = []
        for model_info in models:
            model_id = model_info.modelId

            # flair model is not supported yet
            if "flair" in model_info.tags:
                logging.info(f"Skip flair model: {model_id}.")
                continue

            languages = get_lang_tags(model_info)
            if "en" not in languages and not import_all:
                logging.warning(f"Skip non-English model: {model_id}.")
                continue

            existing_model = self.processed_models.get(model_id)
            if existing_model:
                existing_model["downloads"] = model_info.downloads
                if not args.retry_failed or existing_model[
                        "result"] == "success":
                    logging.info(f"Skip converted model: {model_id}.")
                    continue

            if model_info.downloads < 50 and not import_all:
                logging.info(
                    f"Skip model {model_info.modelId}, downloads {model_info.downloads} < 50"
                )
                continue

            try:
                config = hf_hub_download(repo_id=model_id,
                                         filename="config.json")
            except EnvironmentError:
                logging.info(f"Skip {model_id}, no config.json found.")
                continue

            with open(config) as f:
                config = json.load(f)

            task, architecture = self.to_supported_task(config)
            if not task:
                if "sentence-similarity" in model_info.tags:
                    task = "sentence-similarity"

            if not task:
                logging.info(
                    f"Unsupported model architecture: {architecture} for {model_id}."
                )
                continue

            if args.category and args.category != task:
                logging.info(
                    f"Skip {model_id}, expect task: {args.category}, detected {task}."
                )
                continue

            model = {
                "model_info": model_info,
                "config": config,
                "task": task,
            }
            ret.append(model)

        return ret

    def update_progress(self, model_info: ModelInfo, application: str,
                        result: bool, reason: str, size: int, cpu_only: bool):
        status = {
            "result": "success" if result else "failed",
            "application": application,
            "sha1": model_info.sha,
            "size": size,
            "downloads": model_info.downloads,
        }
        if reason:
            status["reason"] = reason
        if cpu_only:
            status["cpu_only"] = True

        self.processed_models[model_info.modelId] = status

        dict_file = os.path.join(self.output_dir, "models.json")
        with open(dict_file, 'w') as f:
            json.dump(self.processed_models,
                      f,
                      sort_keys=True,
                      indent=2,
                      ensure_ascii=False)

    @staticmethod
    def to_supported_task(config: dict):
        architectures = config.get("architectures")
        if not architectures:
            return None, "No architectures found"

        architecture = architectures[0]
        for arch in ARCHITECTURES_2_TASK:
            if architecture.endswith(arch):
                return ARCHITECTURES_2_TASK[arch], architecture

        return None, architecture
