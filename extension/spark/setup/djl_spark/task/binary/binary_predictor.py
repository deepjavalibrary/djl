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
from typing import Optional


class BinaryPredictor:
    """BinaryPredictor performs prediction on binary input.
    """

    def __init__(self,
                 input_col: str,
                 output_col: str,
                 model_url: str,
                 engine: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 input_class=None,
                 output_class=None,
                 translator_factory=None,
                 batchifier: Optional[str] = None):
        """
        Initializes the BinaryPredictor.

        :param input_col: The input column
        :param output_col: The output column
        :param model_url: The model URL
        :param engine (optional): The engine
        :param batch_size (optional): The batch size
        :param input_class (optional): The input class. Default is byte array.
        :param output_class (optional): The output class. Default is byte array.
        :param translator_factory (optional): The translator factory.
                                              Default is NpBinaryTranslatorFactory.
        :param batchifier (optional): The batchifier. Valid values include "none" (default),
                                      "stack", and "padding".
        """
        self.input_col = input_col
        self.output_col = output_col
        self.model_url = model_url
        self.engine = engine
        self.batch_size = batch_size
        self.input_class = input_class
        self.output_class = output_class
        self.translator_factory = translator_factory
        self.batchifier = batchifier

    def predict(self, dataset):
        """
        Performs prediction on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context
        predictor = sc._jvm.ai.djl.spark.task.binary.BinaryPredictor() \
            .setInputCol(self.input_col) \
            .setOutputCol(self.output_col) \
            .setModelUrl(self.model_url)
        if self.engine is not None:
            predictor = predictor.setEngine(self.engine)
        if self.batch_size is not None:
            predictor = predictor.setBatchSize(self.batch_size)
        if self.input_class is not None:
            predictor = predictor.setinputClass(self.input_class)
        if self.output_class is not None:
            predictor = predictor.setOutputClass(self.output_class)
        if self.translator_factory is not None:
            predictor = predictor.setTranslatorFactory(self.translator_factory)
        if self.batchifier is not None:
            predictor = predictor.setBatchifier(self.batchifier)
        return DataFrame(predictor.predict(dataset._jdf), dataset.sparkSession)
