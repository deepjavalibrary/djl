#!/usr/bin/env python3
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
import os

from setuptools import setup

def read_from_file(file, key):
    with open(file, "r") as f:
        for line in f:
            if not line.startswith('#'):
                prop = line.split("=")
                if prop[0].strip() == key:
                    return prop[1].strip().replace('"', '')
    return None


def detect_version():
    version_file = os.path.join("djl_converter", "__init__.py")
    djl_version = read_from_file(version_file, "__version__")

    if not djl_version:
        current_dir = os.path.dirname(__file__)
        toml_file = f"{current_dir}/../../../../../gradle/libs.versions.toml"
        djl_version = read_from_file(toml_file, "djl")
        with open(version_file, "a") as f:
            f.writelines(['\n', f"__version__ = \"{djl_version}\""])

    return djl_version


if __name__ == '__main__':
    version = detect_version()
    setup(version=version)