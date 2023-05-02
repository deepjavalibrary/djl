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
"""DJL Spark Tasks Text API."""

from . import (
    question_answerer,
    text2text_generator,
    text_classifier,
    text_decoder,
    text_embedder,
    text_encoder,
    text_generator,
    text_tokenizer,
)

QuestionAnswerer = question_answerer.QuestionAnswerer
Text2TextGenerator = text2text_generator.Text2TextGenerator
TextClassifier = text_classifier.TextClassifier
TextDecoder = text_decoder.TextDecoder
TextEmbedder = text_embedder.TextEmbedder
TextEncoder = text_encoder.TextEncoder
TextGenerator = text_generator.TextGenerator
TextTokenizer = text_tokenizer.TextTokenizer

# Remove unnecessary modules to avoid duplication in API.
del (
    question_answerer,
    text2text_generator,
    text_classifier,
    text_decoder,
    text_embedder,
    text_encoder,
    text_generator,
    text_tokenizer,
)
