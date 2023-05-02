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

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import io
import librosa
import pandas as pd
from typing import Iterator, Optional
from transformers import pipeline
from ...util import files_util, dependency_util


TASK = "automatic-speech-recognition"
APPLICATION = "audio/automatic_speech_recognition"
GROUP_ID = "ai/djl/huggingface/pytorch"


class WhisperSpeechRecognizer:

    def __init__(self,
                 input_col: str,
                 output_col: str,
                 model_url: Optional[str] = None,
                 hf_model_id: Optional[str] = None,
                 engine: Optional[str] = "PyTorch",
                 batch_size: Optional[int] = 10):
        """
        Initializes the WhisperSpeechRecognizer.

        :param input_col: The input column
        :param output_col: The output column
        :param model_url: The model URL
        :param hf_model_id: The Huggingface model ID
        :param engine: The engine. Currently only PyTorch is supported.
        :param batch_size: The batch size
        """
        self.input_col = input_col
        self.output_col = output_col
        self.model_url = model_url
        self.hf_model_id = hf_model_id
        self.engine = engine
        self.batch_size = batch_size

    def recognize(self, dataset, generate_kwargs=None, **kwargs):
        """
        Performs speech recognition on the provided dataset.

        :param dataset: input dataset
        :param generate_kwargs: The dictionary of ad-hoc parametrization of generate_config
                                to be used for the generation call.
        :return: output dataset
        """
        if self.engine is None or self.engine.lower() != "pytorch":
            raise ValueError("Only PyTorch engine is supported.")

        if self.model_url:
            cache_dir = files_util.get_cache_dir(APPLICATION, GROUP_ID, self.model_url)
            files_util.download_and_extract(self.model_url, cache_dir)
            dependency_util.install(cache_dir)
            model_id_or_path = cache_dir
        elif self.hf_model_id:
            model_id_or_path = self.hf_model_id
        else:
            raise ValueError("Either model_url or hf_model_id must be provided.")

        @pandas_udf(StringType())
        def predict_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
            pipe = pipeline(TASK, generate_kwargs=generate_kwargs, model=model_id_or_path,
                            batch_size=self.batch_size, chunk_length_s=30, **kwargs)
            for s in iterator:
                # Model expects single channel, 16000 sample rate audio
                batch = [librosa.load(io.BytesIO(d), mono=True, sr=16000)[0] for d in s]
                output = pipe(batch)
                text = [o["text"] for o in output]
                yield pd.Series(text)

        return dataset.withColumn(self.output_col, predict_udf(self.input_col))
