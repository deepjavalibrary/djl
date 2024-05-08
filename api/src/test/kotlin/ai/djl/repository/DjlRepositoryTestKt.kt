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
package ai.djl.repository

import ai.djl.MalformedModelException
import ai.djl.modality.Classifications
import ai.djl.modality.cv.Image
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ModelNotFoundException
import criteria
import org.testng.Assert
import org.testng.annotations.Test
import java.io.IOException

class DjlRepositoryTestKt {
    @Test @Throws(ModelNotFoundException::class, MalformedModelException::class, IOException::class)
    fun testResource() {
        var repo = Repository.newInstance("DJL", "djl://ai.djl.pytorch/resnet")
        Assert.assertEquals(repo.resources.size, 1)

        repo = Repository.newInstance("DJL", "djl://ai.djl.pytorch/resnet/0.0.1")
        Assert.assertEquals(repo.resources.size, 1)

        repo = Repository.newInstance("DJL", "djl://ai.djl.pytorch/resnet/0.0.1/traced_resnet18")
        Assert.assertEquals(repo.resources.size, 1)

        Assert.assertThrows(IllegalArgumentException::class.java) { Repository.newInstance("DJL", "djl://ai.djl.pytorch/fake/0.0.1") }

        Assert.assertThrows(IllegalArgumentException::class.java) { Repository.newInstance("DJL", "djl://ai.djl.fake/mlp/0.0.1") }

        Assert.expectThrows(IllegalArgumentException::class.java) { Repository.newInstance("DJL", "djl://") }

        Assert.expectThrows(IllegalArgumentException::class.java) { Repository.newInstance("DJL", "djl://ai.djl.pytorch") }

        Assert.expectThrows(IllegalArgumentException::class.java) { Repository.newInstance("DJL", "djl://ai.djl.pytorch/") }

        criteria<Image, Classifications> {
            modelUrls = "djl://ai.djl.pytorch/resnet/0.0.1/traced_resnet18"
        }.loadModel().use { model ->
            Assert.assertEquals(model.name, "traced_resnet18")
        }
    }
}
