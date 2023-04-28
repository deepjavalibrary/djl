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


class TextEmbedder:

    def __init__(self, input_col, output_col, model_url, engine=None,
                 output_class=None, translator_factory=None):
        """
        Initializes the TextEmbedder.

        :param input_col: The input column
        :param output_col: The output column
        :param model_url: The model URL
        :param engine (optional): The engine
        :param output_class (optional): The output class
        :param translator_factory (optional): The translator factory. Default is TextEmbeddingTranslatorFactory.
        """
        self.input_col = input_col
        self.output_col = output_col
        self.engine = engine
        self.model_url = model_url
        self.output_class = output_class
        self.translator_factory = translator_factory

    def embed(self, dataset):
        """
        Performs text embedding on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context
        embedder = sc._jvm.ai.djl.spark.task.text.TextEmbedder()
        if self.output_class is not None:
            embedder = embedder.setOutputClass(self.output_class)
        if self.translator_factory is not None:
            embedder = embedder.setTranslatorFactory(self.translator_factory)
        embedder = embedder.setInputCol(self.input_col) \
            .setOutputCol(self.output_col) \
            .setEngine(self.engine) \
            .setModelUrl(self.model_url)
        return DataFrame(embedder.embed(dataset._jdf),
                         dataset.sparkSession)
