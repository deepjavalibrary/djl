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

import argparse
import os


def converter_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", help="Model output directory")
    parser.add_argument("-f",
                        "--output-format",
                        default="PyTorch",
                        choices=["PyTorch", "OnnxRuntime", "Rust"],
                        help="Model output format")
    parser.add_argument("-u",
                        "--cpu-only",
                        action='store_true',
                        help="Only validate jit traced model on CPU")
    parser.add_argument("-m",
                        "--model-id",
                        help="Huggingface model_id to convert")
    parser.add_argument("-r", "--revision", help="Huggingface model revision")
    parser.add_argument("-t", "--token", help="Huggingface token")
    parser.add_argument("--optimize",
                        choices=["O1", "O2", "O3", "O4"],
                        help="Optimization option for ONNX models")
    parser.add_argument(
        "--device",
        help='The device to use to do the export. Defaults to "cpu".')
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        help="The floating point precision to use for the export.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allows to use custom code for the modeling hosted in the model"
        " repository. This option should only be set for repositories you trust and in which"
        " you have read the code, as it will execute on your local machine arbitrary code"
        " present in the model repository.")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "."

    return args


def importer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l",
                        "--limit",
                        type=int,
                        default=1,
                        help="Max amount of models to convert")
    parser.add_argument("-o", "--output-dir", help="Model output directory")
    parser.add_argument("-f",
                        "--output-format",
                        default="PyTorch",
                        choices=["PyTorch", "OnnxRuntime", "Rust"],
                        help="Model output format")
    parser.add_argument("-r",
                        "--retry-failed",
                        action='store_true',
                        help="Retry failed model")
    parser.add_argument("-u",
                        "--cpu-only",
                        action='store_true',
                        help="Only validate jit traced model on CPU")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c",
        "--category",
        help="Model category to convert",
    )
    group.add_argument("-m", "--model-name", help="Model name to convert")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allows to use custom code for the modeling hosted in the model"
        " repository. This option should only be set for repositories you trust and in which"
        " you have read the code, as it will execute on your local machine arbitrary code"
        " present in the model repository.")
    parser.add_argument(
        "--min-version",
        help="Requires a specific version of DJL to load the model.")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "."

    if not os.path.exists(args.output_dir):
        raise ValueError(f"Invalid output directory: {args.output_dir}.")

    return args
