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
package ai.djl.integration.tests.training;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.testing.Assertions;
import java.io.IOException;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ModelTest {

    @Test
    public void testModelSaveAndLoad() throws IOException, MalformedModelException {
        SequentialBlock block = new SequentialBlock();
        block.add(Conv2d.builder().setKernelShape(new Shape(1, 1)).setFilters(10).build());
        block.add(BatchNorm.builder().build());
        try (Model saveModel = Model.newInstance("saveModel");
                Model loadModel = Model.newInstance("loadModel")) {
            block.initialize(saveModel.getNDManager(), DataType.FLOAT32, new Shape(1, 3, 32, 32));
            ParameterList savedParameters = block.getParameters();
            saveModel.setBlock(block);
            saveModel.save(Paths.get("build/tmp/test/models"), "saveAndLoad");
            block.clear();

            loadModel.setBlock(block);
            loadModel.load(Paths.get("build/tmp/test/models"), "saveAndLoad");
            ParameterList loadedParameters = loadModel.getBlock().getParameters();
            compareParameters(savedParameters, loadedParameters);
        }
    }

    private void compareParameters(ParameterList savedParameters, ParameterList loadedParameters) {
        Assert.assertEquals(savedParameters.size(), loadedParameters.size());
        for (int i = 0; i < savedParameters.size(); i++) {
            Assertions.assertAlmostEquals(
                    savedParameters.get(i).getValue().getArray(),
                    loadedParameters.get(i).getValue().getArray());
        }
    }
}
