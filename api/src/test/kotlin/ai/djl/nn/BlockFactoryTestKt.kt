/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn

import ai.djl.Model
import ai.djl.ModelException
import ai.djl.ndarray.NDList
import criteria
import org.testng.Assert
import org.testng.annotations.Test
import java.io.IOException
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap

class BlockFactoryTestKt {
    @Test fun testIdentityBlockFactory() {
        val factory = IdentityBlockFactory()
        Model.newInstance("identity").use { model ->
            val path = Paths.get("build")
            val block = factory.newBlock(model, path, null)
            Assert.assertEquals((block as LambdaBlock).name, "identity")
        }
    }

    @Test
    @Throws(ModelException::class, IOException::class)
    fun testOnesBlockFactory() {
        val factory = OnesBlockFactory()
        val path = Paths.get("build")
        val criteria = criteria<NDList, NDList> {
            modelPath = path
            arguments("blockFactory" to "ai.djl.nn.OnesBlockFactory",
                      "block_shapes" to "(1)s,(1)d,(1)u,(1)b,(1)i,(1)l,(1)B,(1)f,(1)",
                      "block_names" to "1,2,3,4,5,6,7,8,9")
            options("hasParameter" to "false")
        }

        criteria.loadModel().use { model ->
            var block = model.block
            Assert.assertTrue(block is LambdaBlock)

            val args: MutableMap<String, String?> = ConcurrentHashMap()
            args["block_shapes"] = "1,2"
            block = factory.newBlock(model, path, args)
            Assert.assertTrue(block is LambdaBlock)

            args["block_shapes"] = "(1)a"
            Assert.assertThrows { factory.newBlock(model, path, args) }
        }
    }
}
