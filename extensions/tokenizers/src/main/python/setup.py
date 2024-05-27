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

import setuptools.command.build_py
from setuptools import setup, find_packages


def detect_version():
    with open("../../../../../gradle/libs.versions.toml", "r") as f:
        for line in f:
            if not line.startswith('#'):
                prop = line.split('=')
                if prop[0].strip() == "djl":
                    return prop[1].strip().strip('\"')
    return None


class BuildPy(setuptools.command.build_py.build_py):

    def run(self):
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    version = detect_version()

    requirements = [
        'huggingface_hub', 'transformers', 'sentence_transformers', 'torch',
        'optimum[exporters,onnxruntime]', 'safetensors'
    ]

    setup(name='djl_converter',
          version=version,
          description='Model converter utility package',
          author='Deep Java Library team',
          author_email='djl-dev@amazon.com',
          url='https://github.com/deepjavalibrary/djl.git',
          packages=find_packages(exclude=['tests', 'tests.*']),
          cmdclass={
              'build_py': BuildPy,
          },
          install_requires=requirements,
          entry_points={
              'console_scripts': [
                  'djl-import=djl_converter.model_zoo_importer:main',
                  'djl-convert=djl_converter.model_converter:main',
              ]
          },
          include_package_data=True,
          license='Apache License Version 2.0')
