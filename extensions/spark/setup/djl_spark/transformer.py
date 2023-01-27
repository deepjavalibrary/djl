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
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame


class SparkTransformer:

    def __init__(self, input_cols, output_cols, engine, model_url, output_class, translator):
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.engine = engine
        self.model_url = model_url
        self.output_class = output_class
        self.translator = translator

    def transform(self, dataset):
        sc = SparkContext._active_spark_context
        sqlContext = SQLContext(sc)

        # Convert the input_cols to Java array
        input_cols_arr = None
        if self.input_cols is not None:
            input_cols_arr = sc._gateway.new_array(sc._jvm.java.lang.String, len(self.input_cols))
            for i in range(len(self.input_cols)):
                input_cols_arr[i]=self.input_cols[i]

        # Convert the output_cols to Java array
        output_cols_arr = None
        if self.output_cols is not None:
            output_cols_arr = sc._gateway.new_array(sc._jvm.java.lang.String, len(self.output_cols))
            for i in range(len(self.output_cols)):
                output_cols_arr[i]=self.output_cols[i]

        output_clazz = sc._jvm.java.lang.Class.forName(self.output_class)
        transformer = sc._jvm.ai.djl.spark.SparkTransformer()
        if input_cols_arr is not None:
            transformer = transformer.setInputCols(input_cols_arr)
        if output_cols_arr is not None:
            transformer = transformer.setOutputCols(output_cols_arr)
        transformer = transformer.setEngine(self.engine) \
            .setModelUrl(self.model_url) \
            .setOutputClass(output_clazz) \
            .setTranslator(self.translator)
        return DataFrame(transformer .transform(dataset._jdf), sqlContext._ssql_ctx)
