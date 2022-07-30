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
    parser.add_argument("-l",
                        "--limit",
                        type=int,
                        default=1,
                        help="Max amount of models to convert")
    parser.add_argument("-o", "--output-dir", help="Model output directory")
    parser.add_argument("-c",
                        "--categories",
                        nargs="*",
                        help="Model categories to convert")
    args = parser.parse_args()
    if not args.categories:
        args.categories = ["question-answering"]
    if args.output_dir is None:
        args.output_dir = "."

    if not os.path.exists(args.output_dir):
        raise ValueError(f"Invalid output directory: {args.output_dir}.")

    return args
