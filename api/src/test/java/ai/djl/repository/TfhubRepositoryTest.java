/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.repository;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TfhubRepositoryTest {

    @Test
    public void testResource() throws IOException {
        String modelUrl = "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1/";
        Repository repo = Repository.newInstance("tfhub", modelUrl);
        Assert.assertEquals(repo.getResources().size(), 1);
        MRL mrl = repo.getResources().get(0);
        Artifact artifact = repo.resolve(mrl, null);
        for (Artifact.Item item : artifact.getFiles().values()) {
            Assert.assertEquals(item.getExtension(), "tgz");
        }
    }
}
