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

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
from typing import Iterator
from transformers import pipeline


class Text2TextGenerator:

    def __init__(self, input_col, output_col, engine, model_url=None, model_name=None):
        """
        Initializes the Text2TextGenerator.

        :param input_col: The input column
        :param output_col: The output column
        :param engine: The engine. Currently only PyTorch is supported.
        :param model_url: The model URL
        :param model_name: The model name
        """
        self.input_col = input_col
        self.output_col = output_col
        self.engine = engine
        self.model_url = model_url
        self.model_name = model_name

    def generate(self, dataset, **kwargs):
        """
        Performs text2text generation on the provided dataset.

        :param dataset: input dataset
        :return: output dataset
        """
        if not self.model_url and not self.model_name:
            raise ValueError("Either model_url or model_name must be provided.")
        model_name_or_url = self.model_url if self.model_url else self.model_name

        @pandas_udf(StringType())
        def predict_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            generator = pipeline('text2text-generation', model=model_name_or_url, **kwargs)
            for s in iterator:
                output = generator(s.tolist())
                text = [o["generated_text"] for o in output]
                yield pd.Series(text)

        return dataset.withColumn(self.output_col, predict_udf(self.input_col))
