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
package software.amazon.ai;

import java.io.IOException;
import org.apache.mxnet.dataset.Cifar10;
import org.apache.mxnet.engine.MxEngine;
import org.apache.mxnet.zoo.ModelZoo;
import org.testng.annotations.Test;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.test.CoverageUtils;
import software.amazon.ai.zoo.cv.classification.ResNetV1;

public class CoverageTest {

    @Test
    public void test() throws IOException, ClassNotFoundException {
        // API
        CoverageUtils.testGetterSetters(Device.class);

        // model-zoo
        CoverageUtils.testGetterSetters(ResNetV1.class);

        // repository
        CoverageUtils.testGetterSetters(Repository.class);

        // mxnet-dataset
        CoverageUtils.testGetterSetters(Cifar10.class);

        // mxnet-engine
        CoverageUtils.testGetterSetters(MxEngine.class);

        // mxnet-model-zoo
        CoverageUtils.testGetterSetters(ModelZoo.class);
    }
}
