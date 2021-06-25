/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.paddlepaddle.zoo.cv.objectdetection;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A {@link PpFaceDetectionTranslator} that post-process the {@link NDArray} into {@link
 * DetectedObjects} with boundaries.
 */
public class PpFaceDetectionTranslator implements Translator<Image, DetectedObjects> {

    private float shrink;
    private float threshold;
    private List<String> className;

    /**
     * Creates the {@link PpFaceDetectionTranslator} instance.
     *
     * @param arguments the arguments for the translator
     */
    public PpFaceDetectionTranslator(Map<String, ?> arguments) {
        threshold =
                arguments.containsKey("threshold")
                        ? (float) Double.parseDouble(arguments.get("threshold").toString())
                        : 0.7f;
        shrink =
                arguments.containsKey("shrink")
                        ? (float) Double.parseDouble(arguments.get("shrink").toString())
                        : 0.5f;
        className = Arrays.asList("Not Face", "Face");
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager());
        Shape shape = array.getShape();
        array =
                NDImageUtils.resize(
                        array, (int) (shape.get(1) * shrink), (int) (shape.get(0) * shrink));
        array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW RGB -> BGR
        NDArray mean =
                ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(3, 1, 1));
        array = array.sub(mean).mul(0.007843f); // normalization
        array = array.expandDims(0); // make batch dimension
        return new NDList(array);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        NDArray result = list.singletonOrThrow();
        float[] probabilities = result.get(":,1").toFloatArray();
        List<String> names = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        List<BoundingBox> boxes = new ArrayList<>();
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= threshold) {
                float[] array = result.get(i).toFloatArray();
                names.add(className.get((int) array[0]));
                prob.add((double) probabilities[i]);
                boxes.add(
                        new Rectangle(
                                array[2], array[3], array[4] - array[2], array[5] - array[3]));
            }
        }
        return new DetectedObjects(names, prob, boxes);
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}
