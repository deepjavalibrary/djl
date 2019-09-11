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
package org.apache.mxnet.zoo.cv.classification;

import software.amazon.ai.repository.Repository;

public class Resnext extends ImageNetBaseModelLoader {

    private static final String ARTIFACT_ID = "resnext";
    private static final String VERSION = "0.0.1";

    public Resnext(Repository repository) {
        super(repository, ARTIFACT_ID, VERSION);
    }
}
