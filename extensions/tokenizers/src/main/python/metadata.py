#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import json


class HuggingfaceMetadata:

    def __init__(self,
                 artifact_ids: list[str],
                 application: str,
                 translator: str,
                 sha1: str,
                 file_size: int,
                 tags: dict = None):
        self.model_name = artifact_ids[-1]
        artifact_ids[0:-1].insert(0, "ai.djl.huggingface")
        self.group_id = ".".join(artifact_ids)
        self.application = application
        self.translator = translator
        self.sha1 = sha1
        self.file_size = file_size
        self.tags = tags

    def save_metadata(self, metadata_file: str):
        metadata = {
            "metadataVersion": "0.2",
            "resourceType": "model",
            "application": self.application,
            "groupId": self.group_id,
            "artifactId": self.model_name,
            "name": self.model_name,
            "description":
            f"Huggingface transformers model: {self.model_name}",
            "website": "http://www.djl.ai/extensions/tokenizers",
            "licenses": {
                "license": {
                    "name": "The Apache License, Version 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0"
                }
            },
            "artifacts": {
                "version": "0.0.1",
                "snapshot": False,
                "name": self.model_name,
                "properties": self.tags,
                "arguments": {
                    "translatorFactory": self.translator
                },
                "options": {
                    "mapLocation": True
                },
                "files": {
                    "model": {
                        "uri": f"0.0.1/{self.model_name}.zip",
                        "name": "",
                        "sha1Hash": self.sha1,
                        "size": self.file_size
                    }
                }
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata,
                      f,
                      sort_keys=False,
                      indent=2,
                      ensure_ascii=False)
