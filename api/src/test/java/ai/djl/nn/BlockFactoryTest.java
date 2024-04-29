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
package ai.djl.nn;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@SuppressWarnings("PMD.TestClassWithoutTestCases")
public class BlockFactoryTest {

    @Test
    public void testIdentityBlockFactory() {
        IdentityBlockFactory factory = new IdentityBlockFactory();
        try (Model model = Model.newInstance("identity")) {
            Path path = Paths.get("build");
            Block block = factory.newBlock(model, path, null);
            Assert.assertEquals(((LambdaBlock) block).getName(), "identity");
        }
    }

    @Test
    public void testOnesBlockFactory() throws ModelException, IOException {
        OnesBlockFactory factory = new OnesBlockFactory();
        Path path = Paths.get("build");
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(path)
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1)s,(1)d,(1)u,(1)b,(1)i,(1)l,(1)B,(1)f,(1)")
                        .optArgument("block_names", "1,2,3,4,5,6,7,8,9")
                        .optOption("hasParameter", "false")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            Block block = model.getBlock();
            Assert.assertTrue(block instanceof LambdaBlock);

            Map<String, String> args = new ConcurrentHashMap<>();
            args.put("block_shapes", "1,2");
            block = factory.newBlock(model, path, args);
            Assert.assertTrue(block instanceof LambdaBlock);

            args.put("block_shapes", "(1)a");
            Assert.assertThrows(
                    () -> {
                        factory.newBlock(model, path, args);
                    });
        }
    }
}
