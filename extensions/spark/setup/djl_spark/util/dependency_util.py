#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import subprocess
import sys


def install(path):
    """Install a Python package.

    :param path: The path to find the requirements.txt.
    """
    if os.path.exists(os.path.join(path, "requirements.txt")):
        cmd = [
            python_executable(), "-m", "pip", "install", "-r",
            os.path.join(path, "requirements.txt")
        ]
        try:
            subprocess.run(cmd, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            print("Error occurred during installing dependency:", e)


def python_executable():
    """Returns the path of the Python executable, if it exists.

    :return: The path of the Python executable.
    """
    if not sys.executable:
        raise RuntimeError(
            "Failed to retrieve the path of the Python executable.")
    return sys.executable
