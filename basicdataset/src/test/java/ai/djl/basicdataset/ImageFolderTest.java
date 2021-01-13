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
package ai.djl.basicdataset;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ImageFolderTest {

    @Test
    public void testImageFolder() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/imagefolder");
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            ImageFolder dataset =
                    ImageFolder.builder()
                            .setRepository(repository)
                            .optPipeline(
                                    new Pipeline().add(new Resize(100, 100)).add(new ToTensor()))
                            .setSampling(1, false)
                            .build();

            List<String> synsets = Arrays.asList("cat", "dog", "misc");
            Assert.assertEquals(synsets, dataset.getSynset());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray cat =
                        ImageFactory.getInstance()
                                .fromFile(
                                        Paths.get("src/test/resources/imagefolder/cat/kitten.jpg"))
                                .toNDArray(manager);
                NDArray dog =
                        ImageFactory.getInstance()
                                .fromFile(
                                        Paths.get(
                                                "src/test/resources/imagefolder/dog/dog_bike_car.jpg"))
                                .toNDArray(manager);
                NDArray pikachu =
                        ImageFactory.getInstance()
                                .fromFile(
                                        Paths.get(
                                                "src/test/resources/imagefolder/misc/pikachu.png"))
                                .toNDArray(manager);

                Iterator<Batch> ds = trainer.iterateDataset(dataset).iterator();

                Batch catBatch = ds.next();
                Assertions.assertAlmostEquals(
                        catBatch.getData().singletonOrThrow(),
                        NDImageUtils.toTensor(NDImageUtils.resize(cat, 100, 100)).expandDims(0));
                Assert.assertEquals(
                        catBatch.getLabels().singletonOrThrow(), manager.create(new long[] {0}));
                catBatch.close();

                Batch dogBatch = ds.next();
                Assertions.assertAlmostEquals(
                        dogBatch.getData().singletonOrThrow(),
                        NDImageUtils.toTensor(NDImageUtils.resize(dog, 100, 100)).expandDims(0));
                Assert.assertEquals(
                        dogBatch.getLabels().singletonOrThrow(), manager.create(new long[] {1}));
                dogBatch.close();

                Batch pikachuBatch = ds.next();
                Assertions.assertAlmostEquals(
                        pikachuBatch.getData().singletonOrThrow(),
                        NDImageUtils.toTensor(NDImageUtils.resize(pikachu, 100, 100))
                                .expandDims(0));
                Assert.assertEquals(
                        pikachuBatch.getLabels().singletonOrThrow(),
                        manager.create(new long[] {2}));
                pikachuBatch.close();
            }
        }
    }

    @Test
    public void testRandomSplit() throws IOException, TranslateException {
        Repository repository = Repository.newInstance("test", "src/test/resources/imagefolder");
        ImageFolder dataset =
                ImageFolder.builder().setRepository(repository).setSampling(1, false).build();
        RandomAccessDataset[] sets = dataset.randomSplit(75, 25);
        Assert.assertEquals(sets[0].size(), 2);
    }
}
