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


class ImageClassifier:
    """ImageClassifier performs image classification on images.
    """

    def __init__(self,
                 input_cols: list[str],
                 output_col: str,
                 model_url: str,
                 engine: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 translator_factory=None,
                 batchifier: Optional[str] = None,
                 apply_softmax: Optional[bool] = None,
                 top_k: Optional[int] = None):
        """
        Initializes the ImageClassifier.

        :param input_cols: The input columns
        :param output_col: The output column
        :param model_url: The model URL
        :param engine (optional): The engine
        :param batch_size (optional): The batch size
        :param translator_factory (optional): The translator factory.
                                              Default is ImageClassificationTranslatorFactory.
        :param batchifier (optional): The batchifier. Valid values include "none" (default),
                                      "stack", and "padding".
        :param apply_softmax (optional): Whether to apply softmax when processing output.
        :param top_k (optional): The number of classes to return.
        """
        self.input_cols = input_cols
        self.output_col = output_col
        self.model_url = model_url
        self.engine = engine
        self.batch_size = batch_size
        self.translator_factory = translator_factory
        self.batchifier = batchifier
        self.apply_softmax = apply_softmax
        self.top_k = top_k

    def classify(self, dataset):
        """
        Performs image classification on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context
        classifier = sc._jvm.ai.djl.spark.task.vision.ImageClassifier() \
            .setOutputCol(self.output_col) \
            .setModelUrl(self.model_url)
        if self.input_cols is not None:
            # Convert the input_cols to Java array
            input_cols_arr = sc._gateway.new_array(sc._jvm.java.lang.String,
                                                   len(self.input_cols))
            input_cols_arr[:] = [col for col in self.input_cols]
            classifier = classifier.setInputCols(input_cols_arr)
        if self.engine is not None:
            classifier = classifier.setEngine(self.engine)
        if self.batch_size is not None:
            classifier = classifier.setBatchSize(self.batch_size)
        if self.translator_factory is not None:
            classifier = classifier.setTranslatorFactory(
                self.translator_factory)
        if self.batchifier is not None:
            classifier = classifier.setBatchifier(self.batchifier)
        if self.apply_softmax is not None:
            classifier = classifier.setApplySoftmax(self.apply_softmax)
        if self.top_k is not None:
            classifier = classifier.setTopK(self.top_k)
        return DataFrame(classifier.classify(dataset._jdf),
                         dataset.sparkSession)
