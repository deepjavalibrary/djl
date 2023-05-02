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

from pyspark import SparkContext
from pyspark.sql import DataFrame


class TextTokenizer:

    def __init__(self,
                 input_col: str,
                 output_col: str,
                 hf_model_id: str):
        """
        Initializes the TextTokenizer.

        :param input_col: The input column
        :param output_col: The output column
        :param hf_model_id: The Huggingface model ID
        """
        self.input_col = input_col
        self.output_col = output_col
        self.hf_model_id = hf_model_id

    def tokenize(self, dataset):
        """
        Performs sentence tokenization on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context
        tokenizer = (
            sc._jvm.ai.djl.spark.task.text.TextTokenizer()
            .setInputCol(self.input_col)
            .setOutputCol(self.output_col)
            .setHfModelId(self.hf_model_id)
        )
        return DataFrame(tokenizer.tokenize(dataset._jdf), dataset.sparkSession)
