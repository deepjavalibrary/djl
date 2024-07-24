#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import sys

from huggingface_hub import HfApi

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from djl_converter.arg_parser import converter_args


def main():
    logging.basicConfig(stream=sys.stdout,
                        format="%(message)s",
                        level=logging.INFO)
    args = converter_args()

    output_dir = args.output_dir
    if output_dir == ".":
        output_dir = f"model/{args.model_id.split('/')[-1]}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.listdir(output_dir):
        logging.error(f"output directory: {output_dir} is not empty.")
        return

    api = HfApi()
    model_info = api.model_info(args.model_id,
                                revision=args.revision,
                                token=args.token)

    from djl_converter.huggingface_models import HuggingfaceModels, SUPPORTED_TASKS

    task, arch = HuggingfaceModels.to_supported_task(model_info.config)
    if not task:
        if "sentence-similarity" in model_info.tags:
            task = "sentence-similarity"
    if not task:
        logging.error(
            f"Unsupported model architecture: {arch} for {model_id}.")
        return

    converter = SUPPORTED_TASKS[task]

    try:
        result, reason, _ = converter.save_model(model_info, args, output_dir,
                                                 False)
        if result:
            logging.info(f"Convert model {model_info.modelId} finished.")
        else:
            logging.error(f"{model_info.modelId}: {reason}")
    except Exception as e:
        logging.warning(f"Failed to convert model: {model_info.modelId}.")
        logging.warning(e, exc_info=True)


if __name__ == "__main__":
    main()
