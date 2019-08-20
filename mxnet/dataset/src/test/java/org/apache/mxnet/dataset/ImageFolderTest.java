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
package org.apache.mxnet.dataset;

import org.apache.mxnet.engine.MxImages;
import org.testng.annotations.Test;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;

public class ImageFolderTest {

    @Test
    public void testImageFolder() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageFolder dataset =
                    new ImageFolder.Builder(manager)
                            .setPath("src/test/resources/imagefolder")
                            .build();
            NDArray cat = MxImages.read(manager, "src/test/resources/imagefolder/cat/cat2.jpeg");
            NDArray dog = MxImages.read(manager, "src/test/resources/imagefolder/dog/puppy1.jpg");
            Assertions.assertAlmostEquals(cat, dataset.get(0).getKey().head());
            Assertions.assertEquals(manager.create(0), dataset.get(0).getValue().head());
            Assertions.assertAlmostEquals(dog, dataset.get(1).getKey().head());
            Assertions.assertEquals(manager.create(1), dataset.get(1).getValue().head());
        }
    }
}
