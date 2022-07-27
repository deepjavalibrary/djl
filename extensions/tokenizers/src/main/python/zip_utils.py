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
import os
import zipfile


def add_to_zip(path: str, handle: zipfile.ZipFile):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            entry_name = os.path.relpath(file_path, path)
            handle.write(file_path, entry_name)


def zip_dir(src_dir: str, output_file: str):
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as f:
        add_to_zip(src_dir, f)
