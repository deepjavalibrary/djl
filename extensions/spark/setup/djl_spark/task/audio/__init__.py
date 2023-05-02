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

"""DJL Spark Tasks Audio API."""

from . import (
    speech_recognizer,
    whisper_speech_recognizer,
)

SpeechRecognizer = speech_recognizer.SpeechRecognizer
WhisperSpeechRecognizer = whisper_speech_recognizer.WhisperSpeechRecognizer

# Remove unnecessary modules to avoid duplication in API.
del (
    speech_recognizer,
    whisper_speech_recognizer,
)
