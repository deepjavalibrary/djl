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
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
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

import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Regression test for GitHub issue #3794: {@link Sam2Translator} only ever returned the
 * highest-scoring mask, with no way to get the full set of candidate masks.
 */
public class Sam2TranslatorTest {

    @Test
    public void testDefaultReturnsOnlyBestMask()
            throws ModelException, IOException, TranslateException {
        DetectedObjects result = predict(false);

        Assert.assertEquals(result.getNumberOfObjects(), 1);
        Assert.assertEquals(result.item(0).getProbability(), 0.9, 1e-6);
    }

    @Test
    public void testMultimaskOutputReturnsAllCandidates()
            throws ModelException, IOException, TranslateException {
        DetectedObjects result = predict(true);

        Assert.assertEquals(result.getNumberOfObjects(), 3);
        Assert.assertEquals(result.item(0).getProbability(), 0.2, 1e-6);
        Assert.assertEquals(result.item(1).getProbability(), 0.9, 1e-6);
        Assert.assertEquals(result.item(2).getProbability(), 0.1, 1e-6);

        // Each candidate's logits fixture is uniformly -1, 1, or -1 respectively, so a pixel's
        // probability distinguishes which raw mask actually made it into each returned object -
        // this fails if every entry carries the same (e.g. best-scoring) mask.
        DetectedObjects.DetectedObject item0 = result.item(0);
        DetectedObjects.DetectedObject item1 = result.item(1);
        DetectedObjects.DetectedObject item2 = result.item(2);
        Assert.assertEquals(((Mask) item0.getBoundingBox()).getProbDist()[0][0], 0f, 1e-6);
        Assert.assertEquals(((Mask) item1.getBoundingBox()).getProbDist()[0][0], 1f, 1e-6);
        Assert.assertEquals(((Mask) item2.getBoundingBox()).getProbDist()[0][0], 0f, 1e-6);
    }

    private DetectedObjects predict(boolean multimaskOutput)
            throws ModelException, IOException, TranslateException {
        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            // 3 candidate masks, 4x4, with a leading dim of 1 matching the raw
                            // (unbatched) model output shape that Sam2Translator itself squeezes.
                            NDArray logits =
                                    manager.create(
                                            new float[] {
                                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, -1, -1
                                            },
                                            new Shape(1, 3, 4, 4));
                            NDArray scores =
                                    manager.create(new float[] {0.2f, 0.9f, 0.1f}, new Shape(1, 3));
                            return new NDList(logits, scores);
                        },
                        "model");

        BufferedImage buf = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        Image img = ImageFactory.getInstance().fromImage(buf);

        Sam2Translator translator =
                Sam2Translator.builder().optMultimaskOutput(multimaskOutput).build();

        Criteria<Sam2Input, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Sam2Input.class, DetectedObjects.class)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optOption("hasParameter", "false")
                        .optTranslator(translator)
                        .build();

        try (ZooModel<Sam2Input, DetectedObjects> model = criteria.loadModel();
                Predictor<Sam2Input, DetectedObjects> predictor = model.newPredictor()) {
            Sam2Input input = Sam2Input.builder(img).addPoint(1, 1).build();
            return predictor.predict(input);
        }
    }
}
