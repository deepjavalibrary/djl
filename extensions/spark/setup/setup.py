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

import setuptools.command.build_py
from setuptools import setup, find_packages

pkgs = find_packages(exclude='src')


def detect_version():
    with open("../../../gradle/libs.versions.toml", "r") as f:
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
        'packaging', 'wheel', 'pillow', 'pandas', 'numpy', 'pyarrow'
    ]

    test_requirements = ['numpy', 'requests']

    setup(name='djl_spark',
          version=version,
          description='djl_spark is a DJL extension to support Spark',
          author='Deep Java Library team',
          author_email='djl-dev@amazon.com',
          long_description=open('PyPiDescription.rst').read(),
          url='https://github.com/deepjavalibrary/djl.git',
          keywords='DJL Spark',
          packages=pkgs,
          cmdclass={
              'build_py': BuildPy,
          },
          install_requires=requirements,
          extras_require={'test': test_requirements + requirements},
          include_package_data=True,
          license='Apache License Version 2.0')
