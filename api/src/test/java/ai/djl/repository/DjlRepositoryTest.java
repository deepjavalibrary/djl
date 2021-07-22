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

import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DjlRepositoryTest {

    @Test
    public void testResource() throws ModelNotFoundException, MalformedModelException, IOException {
        Repository repo = Repository.newInstance("DJL", "djl://ai.djl.mxnet/mlp");
        Assert.assertEquals(repo.getResources().size(), 1);

        repo = Repository.newInstance("DJL", "djl://ai.djl.mxnet/resnet/0.0.1");
        Assert.assertEquals(repo.getResources().size(), 1);

        repo = Repository.newInstance("DJL", "djl://ai.djl.mxnet/resnet/0.0.1/resnet18_v1");
        Assert.assertEquals(repo.getResources().size(), 1);

        repo = Repository.newInstance("DJL", "djl://ai.djl.mxnet/fake/0.0.1");
        Assert.assertEquals(repo.getResources().size(), 0);

        repo = Repository.newInstance("DJL", "djl://ai.djl.fake/mlp/0.0.1");
        Assert.assertEquals(repo.getResources().size(), 0);

        Assert.expectThrows(
                IllegalArgumentException.class, () -> Repository.newInstance("DJL", "djl://"));

        Assert.expectThrows(
                IllegalArgumentException.class,
                () -> Repository.newInstance("DJL", "djl://ai.djl.mxnet"));

        Assert.expectThrows(
                IllegalArgumentException.class,
                () -> Repository.newInstance("DJL", "djl://ai.djl.mxnet/"));

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelUrls("djl://ai.djl.mxnet/resnet/0.0.1/resnet18_v1")
                        .build();
        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            Assert.assertEquals(model.getName(), "resnet18_v1");
        }
    }
}
