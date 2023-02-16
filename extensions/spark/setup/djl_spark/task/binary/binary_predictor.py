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
from pyspark.sql import DataFrame, SparkSession


class BinaryPredictor:
    """BinaryPredictor performs prediction on binary input.
    """

    def __init__(self, input_col, output_col, model_url, engine=None,
                 input_class=None, output_class=None, translator=None,
                 batchifier="none"):
        """
        Initializes the BinaryPredictor.

        :param input_col: The input column.
        :param output_col: The output column.
        :param model_url: The model URL.
        :param engine (optional): The engine.
        :param input_class (optional): The input class. Default is byte array.
        :param output_class (optional): The output class. Default is byte array.
        :param translator (optional): The translator. Default is NpBinaryTranslator.
        :param batchifier (optional): The batchifier. Valid values include none (default),
                                      stack, and padding.
        """
        self.input_col = input_col
        self.output_col = output_col
        self.model_url = model_url
        self.engine = engine
        self.input_class = input_class
        self.output_class = output_class
        self.translator = translator
        self.batchifier = batchifier

    def predict(self, dataset):
        """
        Performs prediction on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context

        predictor = sc._jvm.ai.djl.spark.task.binary.BinaryPredictor()
        if self.input_class is not None:
            predictor = predictor.setinputClass(self.input_class)
        if self.output_class is not None:
            predictor = predictor.setOutputClass(self.output_class)
        if self.translator is not None:
            self.translator = predictor.setTranslator(self.translator)
        predictor = predictor.setInputCol(self.input_col) \
            .setOutputCol(self.output_col) \
            .setModelUrl(self.model_url) \
            .setEngine(self.engine) \
            .setBatchifier(self.batchifier)
        return DataFrame(predictor.predict(dataset._jdf),
                         dataset.sparkSession)
