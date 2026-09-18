/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Regression test for the MXNet SSD zoo pixel-coordinate bug (GitHub issue #3889): a model whose
 * zoo metadata sets {@code rescale} but not {@code applyRatio} must still get its raw pixel boxes
 * normalized to width/height ratios.
 */
public class SingleShotDetectionTranslatorTest {

    private static final int IMAGE_SIZE = 512;

    @Test
    public void testRescaleArgumentNormalizesPixelBoxes()
            throws ModelException, IOException, TranslateException {
        DetectedObjects.DetectedObject detection =
                predictWithArguments(true, false).best();

        BoundingBox box = detection.getBoundingBox();
        // The fake model below emits a pixel-space box of (100, 100, 200, 200) out of a
        // 512x512 input. With "rescale" honored, that must come back as a 0..1 ratio.
        Assert.assertTrue(box.getBounds().getX() < 1.0, "x should be a ratio, not a pixel value");
        Assert.assertTrue(box.getBounds().getY() < 1.0, "y should be a ratio, not a pixel value");
        Assert.assertEquals(box.getBounds().getX(), 100.0 / IMAGE_SIZE, 1e-6);
        Assert.assertEquals(box.getBounds().getY(), 100.0 / IMAGE_SIZE, 1e-6);
        Assert.assertEquals(box.getBounds().getWidth(), 100.0 / IMAGE_SIZE, 1e-6);
        Assert.assertEquals(box.getBounds().getHeight(), 100.0 / IMAGE_SIZE, 1e-6);
    }

    @Test
    public void testWithoutRescaleOrApplyRatioKeepsRawValues()
            throws ModelException, IOException, TranslateException {
        // Baseline: a model that already emits ratios (no rescale, no applyRatio) must be passed
        // through unchanged -- this is the behavior #3841 fixed and must not regress.
        DetectedObjects.DetectedObject detection =
                predictWithArguments(false, false).best();

        BoundingBox box = detection.getBoundingBox();
        Assert.assertEquals(box.getBounds().getX(), 100.0, 1e-6);
        Assert.assertEquals(box.getBounds().getY(), 100.0, 1e-6);
    }

    @Test
    public void testApplyRatioArgumentStillWorks()
            throws ModelException, IOException, TranslateException {
        DetectedObjects.DetectedObject detection =
                predictWithArguments(false, true).best();

        BoundingBox box = detection.getBoundingBox();
        Assert.assertEquals(box.getBounds().getX(), 100.0 / IMAGE_SIZE, 1e-6);
        Assert.assertEquals(box.getBounds().getY(), 100.0 / IMAGE_SIZE, 1e-6);
    }

    private DetectedObjects predictWithArguments(boolean rescale, boolean applyRatio)
            throws ModelException, IOException, TranslateException {
        Block block =
                new LambdaBlock(
                        a -> {
                            // Shapes carry a leading batch dimension (size 1), matching a real
                            // SSD model's output for a single-image batch: the batchifier slices
                            // that dimension off before processOutput sees these NDArrays.
                            NDManager manager = a.getManager();
                            NDArray classIds = manager.create(new float[] {0}, new Shape(1, 1));
                            NDArray probabilities =
                                    manager.create(new float[] {0.9f}, new Shape(1, 1));
                            NDArray boundingBoxes =
                                    manager.create(
                                            new float[] {100, 100, 200, 200}, new Shape(1, 1, 4));
                            return new NDList(classIds, probabilities, boundingBoxes);
                        },
                        "model");

        Path imageFile = Paths.get("../examples/src/test/resources/kitten.jpg");
        Image img = ai.djl.modality.cv.ImageFactory.getInstance().fromFile(imageFile);

        Criteria.Builder<Image, DetectedObjects> criteriaBuilder =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("width", IMAGE_SIZE)
                        .optArgument("height", IMAGE_SIZE)
                        .optArgument("synset", "cat")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new SingleShotDetectionTranslatorFactory());
        if (rescale) {
            criteriaBuilder.optArgument("rescale", true);
        }
        if (applyRatio) {
            criteriaBuilder.optArgument("applyRatio", true);
        }

        try (ZooModel<Image, DetectedObjects> model = criteriaBuilder.build().loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }
}
