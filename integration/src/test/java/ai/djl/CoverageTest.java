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
package ai.djl;

import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.repository.Repository;
import ai.djl.testing.CoverageUtils;
import java.io.IOException;
import java.net.URISyntaxException;
import org.testng.annotations.Test;

public class CoverageTest {

    @Test
    public void test() throws IOException, ReflectiveOperationException, URISyntaxException {
        // API
        CoverageUtils.testGetterSetters(Device.class);

        // model-zoo
        CoverageUtils.testGetterSetters(ResNetV1.class);

        // repository
        CoverageUtils.testGetterSetters(Repository.class);

        // basicdataset
        CoverageUtils.testGetterSetters(Cifar10.class);
    }
}
