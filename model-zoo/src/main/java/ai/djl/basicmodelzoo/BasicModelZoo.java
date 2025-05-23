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
package ai.djl.basicmodelzoo;

import ai.djl.Application.CV;
import ai.djl.repository.RemoteRepository;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;

import java.util.HashSet;
import java.util.Set;

/** Model Zoo is a repository that contains all models for DJL. */
public class BasicModelZoo extends ModelZoo {

    private static final Repository REPOSITORY = new RemoteRepository("zoo", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.zoo";

    BasicModelZoo() {
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mlp", "0.0.3"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.2"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.2"));
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        Set<String> set = new HashSet<>();
        set.add("MXNet");
        set.add("PyTorch");
        // TODO Currently WIP in supporting these two engines in the basic model zoo
        //        set.add("TensorFlow");
        return set;
    }
}
