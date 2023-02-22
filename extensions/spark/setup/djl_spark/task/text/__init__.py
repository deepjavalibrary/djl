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

from . import huggingface_text_decoder
from . import huggingface_text_encoder
from . import huggingface_text_tokenizer
from . import text_embedder

HuggingFaceTextDecoder = huggingface_text_decoder.HuggingFaceTextDecoder
HuggingFaceTextEncoder = huggingface_text_encoder.HuggingFaceTextEncoder
HuggingFaceTextTokenizer = huggingface_text_tokenizer.HuggingFaceTextTokenizer
TextEmbedder = text_embedder.TextEmbedder

# Remove unnecessary modules to avoid duplication in API.
del huggingface_text_decoder
del huggingface_text_encoder
del huggingface_text_tokenizer
del text_embedder