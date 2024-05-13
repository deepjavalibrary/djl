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
import shutil
import sys

from djl_converter.arg_parser import converter_args
from djl_converter.fill_mask_converter import FillMaskConverter
from djl_converter.huggingface_models import HuggingfaceModels
from djl_converter.question_answering_converter import QuestionAnsweringConverter
from djl_converter.sentence_similarity_converter import SentenceSimilarityConverter
from djl_converter.text_classification_converter import TextClassificationConverter
from djl_converter.token_classification_converter import TokenClassificationConverter

SUPPORTED_TASK = {
    "fill-mask": FillMaskConverter(),
    "question-answering": QuestionAnsweringConverter(),
    "sentence-similarity": SentenceSimilarityConverter(),
    "text-classification": TextClassificationConverter(),
    "token-classification": TokenClassificationConverter(),
}


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    args = converter_args()

    huggingface_models = HuggingfaceModels(args.output_dir)
    temp_dir = f"{args.output_dir}/tmp"

    models = huggingface_models.list_models(args)
    if not models:
        logging.warning(f"model not found: {args}")

    for model in models:
        task = model["task"]
        model_info = model["model_info"]
        converter = SUPPORTED_TASK[task]

        try:
            result, reason, size = converter.save_model(
                model_info, args, temp_dir)
            if not result:
                logging.error(f"{model_info.modelId}: {reason}")
        except Exception as e:
            logging.warning(f"Failed to convert model: {model_info.modelId}.")
            logging.warning(e, exc_info=True)
            result = False
            reason = "Failed to convert model"
            size = -1

        huggingface_models.update_progress(model_info, converter.application,
                                           result, reason, size, args.cpu_only)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    logging.info("finished.")


if __name__ == "__main__":
    main()
