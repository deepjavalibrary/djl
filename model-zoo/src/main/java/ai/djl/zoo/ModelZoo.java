/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.zoo;

import ai.djl.repository.Repository;
import ai.djl.zoo.cv.classification.MlpModelLoader;
import ai.djl.zoo.cv.classification.ResNetModelLoader;

public interface ModelZoo {

    String REPO_URL = "https://djl-ai.s3.amazonaws.com/mlrepo/";
    Repository REPOSITORY = Repository.newInstance("DjlRepo", REPO_URL);
    String GROUP_ID = "ai.djl.zoo";

    ResNetModelLoader RESNET = new ResNetModelLoader(REPOSITORY);
    MlpModelLoader MLP = new MlpModelLoader(REPOSITORY);
}
