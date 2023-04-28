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

from . import text_decoder, text_encoder, text_tokenizer, text_embedder, text2text_generator, text_generator

TextDecoder = text_decoder.TextDecoder
TextEncoder = text_encoder.TextEncoder
TextTokenizer = text_tokenizer.TextTokenizer
TextEmbedder = text_embedder.TextEmbedder
Text2TextGenerator = text2text_generator.Text2TextGenerator
TextGenerator = text_generator.TextGenerator

# Remove unnecessary modules to avoid duplication in API.
del text_decoder
del text_encoder
del text_tokenizer
del text_embedder
del text2text_generator
del text_generator