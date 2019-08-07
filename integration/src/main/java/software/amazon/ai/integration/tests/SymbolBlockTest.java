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
package software.amazon.ai.integration.tests;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.mxnet.engine.SymbolBlock;
import software.amazon.ai.Block;
import software.amazon.ai.Model;
import software.amazon.ai.SequentialBlock;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.MnistUtils;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.initializer.NormalInitializer;

// TODO: This test should be replaced in examples for
// Inference tutorial, Transfer Learning and fine tuning
public class SymbolBlockTest {
    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", SymbolBlockTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testInference() throws FailedTestException, IOException {
        Path modelPathPrefix = Paths.get(MnistUtils.prepareModel() + "/mnist");
        Model mnistmlp = Model.loadModel(modelPathPrefix);
        Block block = mnistmlp.getBlock();
        Shape shape;
        try (NDManager manager = NDManager.newBaseManager().newSubManager()) {
            NDArray arr = manager.ones(new Shape(1, 28, 28));
            shape = block.forward(new NDList(arr)).head().getShape();
        }
        mnistmlp.close();
        Assertions.assertTrue(shape.equals(new Shape(1, 10)));
    }

    @RunAsTest
    public void trainWithNewParam() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(MnistUtils.prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            Block mlp = mnistmlp.getBlock();
            mlp.setInitializer(manager, new NormalInitializer(0.01));
            MnistUtils.trainMnist(mlp, manager, 2, 0.5f, 0.8f);
            mnistmlp.close();
        }
    }

    @RunAsTest
    public void trainWithExistParam() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(MnistUtils.prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            Block mlp = mnistmlp.getBlock();
            ((SymbolBlock) mlp).setForTraining();
            MnistUtils.trainMnist(mlp, manager, 1, 0.1f, 0.9f);
            mnistmlp.close();
        }
    }

    @RunAsTest
    public void trainWithCustomLayer() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(MnistUtils.prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            SymbolBlock mlp = (SymbolBlock) mnistmlp.getBlock();
            SymbolBlock mlpPartial = mlp.getLayer("hybridsequential0_dense1_relu_fwd_output");
            SequentialBlock newMlp = new SequentialBlock();
            newMlp.add(mlpPartial);
            newMlp.add(new Linear.Builder().setOutChannels(10).build());
            newMlp.setInitializer(manager, new NormalInitializer(0.01));
            MnistUtils.trainMnist(newMlp, manager, 2, 0.52f, 0.8f);
            mnistmlp.close();
        }
    }
}
