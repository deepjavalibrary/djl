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


class ImageClassifier:
    """ImageClassifier performs image classification on images.
    """

    def __init__(self, input_cols, output_col, engine, model_url,
                 output_class=None, translator=None, topK=5):
        """
        Initializes the ImageClassifier.

        :param input_cols: The input columns
        :param output_col: The output column
        :param engine (optional): The engine
        :param model_url: The model URL
        :param output_class (optional): The output class
        :param translator (optional): The translator. Default is ImageClassificationTranslator.
        :param topK (optional): The number of classes to return. Default is 5.
        """
        self.input_cols = input_cols
        self.output_col = output_col
        self.engine = engine
        self.model_url = model_url
        self.output_class = output_class
        self.translator = translator
        self.topK = topK

    def classify(self, dataset):
        """
        Performs image classification on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        sc = SparkContext._active_spark_context

        # Convert the input_cols to Java array
        input_cols_arr = None
        if self.input_cols is not None:
            input_cols_arr = sc._gateway.new_array(sc._jvm.java.lang.String,
                                                   len(self.input_cols))
            for i in range(len(self.input_cols)):
                input_cols_arr[i] = self.input_cols[i]

        classifier = sc._jvm.ai.djl.spark.task.vision.ImageClassifier()
        if input_cols_arr is not None:
            classifier = classifier.setInputCols(input_cols_arr)
        if self.output_class is not None:
            classifier = classifier.setOutputClass(self.output_class)
        if self.translator is not None:
            classifier = classifier.setTranslator(self.translator)
        classifier = classifier.setOutputCol(self.output_col) \
            .setEngine(self.engine) \
            .setModelUrl(self.model_url) \
            .setTopK(self.topK)
        return DataFrame(classifier.classify(dataset._jdf),
                         dataset.sparkSession)
