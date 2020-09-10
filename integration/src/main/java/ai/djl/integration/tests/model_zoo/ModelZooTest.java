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
package ai.djl.integration.tests.model_zoo;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.cv.classification.AlexNet;
import ai.djl.integration.util.TestUtils;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooProvider;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.ServiceLoader;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class ModelZooTest {

    @BeforeClass
    public void setUp() {
        // force downloading without cache in .djl.ai folder.
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        String userHome = System.getProperty("user.home");
        System.setProperty("ENGINE_CACHE_DIR", userHome);
    }

    @AfterClass
    public void tearDown() {
        System.setProperty("DJL_CACHE_DIR", "");
    }

    @Test
    public void testDownloadModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        ServiceLoader<ZooProvider> providers = ServiceLoader.load(ZooProvider.class);
        for (ZooProvider provider : providers) {
            ModelZoo zoo = provider.getModelZoo();
            if (zoo != null) {
                for (ModelLoader modelLoader : zoo.getModelLoaders()) {
                    List<Artifact> artifacts = modelLoader.listModels();
                    for (Artifact artifact : artifacts) {
                        Criteria<NDList, NDList> criteria =
                                Criteria.builder()
                                        .setTypes(NDList.class, NDList.class)
                                        .optFilters(artifact.getProperties())
                                        .build();
                        Model model = modelLoader.loadModel(criteria);
                        model.close();
                    }
                    Utils.deleteQuietly(Paths.get("build/cache"));
                }
            }
        }
    }

    @Test
    public void testImperativeModelInputOutput()
            throws MalformedModelException, ModelNotFoundException, IOException {
        // Test imperative models, only available in MXNet engine
        if (!TestUtils.isMxnet()) {
            throw new SkipException("Resnet50-cifar10 model only available in MXNet");
        }
        // from model zoo
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optGroupId(BasicModelZoo.GROUP_ID)
                        .optArtifactId("resnet")
                        .optFilter("layers", "50")
                        .optFilter("dataset", "cifar10")
                        .build();
        try (Model model = ModelZoo.loadModel(criteria)) {
            Assert.assertEquals(model.describeInput().values().get(0), new Shape(1, 3, 32, 32));
            Assert.assertEquals(model.describeOutput().values().get(0), new Shape(1, 10));
        }

        Criteria<Image, DetectedObjects> ssdCriteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optGroupId(BasicModelZoo.GROUP_ID)
                        .build();
        try (Model model = ModelZoo.loadModel(ssdCriteria)) {
            Assert.assertEquals(model.describeInput().values().get(0), new Shape(32, 3, 256, 256));
            Assert.assertEquals(model.describeOutput().values().get(0), new Shape(1, 5444, 4));
            Assert.assertEquals(model.describeOutput().values().get(1), new Shape(32, 5444, 2));
            Assert.assertEquals(model.describeOutput().values().get(2), new Shape(32, 21776));
        }

        // from builder
        Block alexNet = AlexNet.builder().build();
        try (Model model = Model.newInstance("alexnet")) {
            model.setBlock(alexNet);
            try (Trainer trainer =
                    model.newTrainer(new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss()))) {
                Shape inputShape = new Shape(32, 3, 224, 224);
                trainer.initialize(inputShape);
                Assert.assertEquals(
                        model.describeInput().values().get(0), new Shape(32, 3, 224, 224));
                Assert.assertEquals(model.describeOutput().values().get(0), new Shape(32, 10));
            }
        }
    }
}
