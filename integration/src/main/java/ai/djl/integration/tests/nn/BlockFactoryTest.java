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
package ai.djl.integration.tests.nn;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.ParameterStore;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BlockFactoryTest {

    @Test
    public void testBlockFactoryLoadingFromZip()
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {
        Path savedDir = Paths.get("build/testBlockFactory");
        Utils.deleteQuietly(savedDir);
        Path zipPath;
        try {
            zipPath = prepareModel(savedDir);
        } catch (ModelNotFoundException e) {
            throw new UnsupportedOperationException(
                    "No test model for engine: " + Engine.getInstance().getEngineName(), e);
        }
        // load model from here
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(zipPath)
                        .optModelName("exported")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> pred = model.newPredictor()) {
            NDManager manager = model.getNDManager();
            NDList destOut = pred.predict(new NDList(manager.ones(new Shape(1, 3, 32, 32))));
            Assert.assertEquals(destOut.singletonOrThrow().getShape(), new Shape(1, 10));
        }
    }

    private Path prepareModel(Path savedDir)
            throws IOException, ModelNotFoundException, MalformedModelException {
        TestBlockFactory factory = new TestBlockFactory();
        try (Model model = factory.getRemoveLastBlockModel();
                NDManager manager = NDManager.newBaseManager()) {
            Block block = model.getBlock();
            block.forward(
                    new ParameterStore(manager, true),
                    new NDList(manager.ones(new Shape(1, 3, 32, 32))),
                    true);
            model.save(savedDir, "exported");
        }
        Path classDir = savedDir.resolve("classes/ai/djl/integration/tests/nn");
        Files.createDirectories(classDir);
        Files.copy(
                Paths.get(
                        "build/classes/java/main/ai/djl/integration/tests/nn/BlockFactoryTest$TestBlockFactory.class"),
                classDir.resolve("BlockFactoryTest$TestBlockFactory.class"));
        Path zipPath =
                Paths.get("build/testBlockFactory" + Engine.getInstance().getEngineName() + ".zip");
        Files.deleteIfExists(zipPath);
        ZipUtils.zip(savedDir, zipPath, false);
        return zipPath;
    }

    public static class TestBlockFactory implements BlockFactory {

        private static final long serialVersionUID = 1234567L;

        @Override
        public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
            SequentialBlock newBlock = new SequentialBlock();
            newBlock.add(SymbolBlock.newInstance(model.getNDManager()));
            newBlock.add(Linear.builder().setUnits(10).build());
            return newBlock;
        }

        public Model getRemoveLastBlockModel()
                throws MalformedModelException, ModelNotFoundException, IOException {
            String name = Engine.getInstance().getEngineName();
            Criteria.Builder<Image, Classifications> builder =
                    Criteria.builder()
                            .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                            .setTypes(Image.class, Classifications.class)
                            .optProgress(new ProgressBar())
                            .optArtifactId("resnet")
                            .optEngine(name)
                            .optGroupId("ai.djl." + name.toLowerCase())
                            .optFilter("layers", "50")
                            .optFilter("flavor", "v2");
            Model model = builder.build().loadModel();
            SequentialBlock newBlock = new SequentialBlock();
            SymbolBlock block = (SymbolBlock) model.getBlock();
            newBlock.add(block);
            newBlock.add(Linear.builder().setUnits(10).build());
            model.setBlock(newBlock);
            return model;
        }
    }
}
