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
package ai.djl.paddlepaddle.zoo.cv.objectdetection;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * A {@link Translator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries.
 */
public class PpWordDetectionTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {

    private final int maxLength;

    /**
     * Creates the {@link PpWordDetectionTranslator} instance.
     *
     * @param arguments the arguments for the translator
     */
    public PpWordDetectionTranslator(Map<String, ?> arguments) {
        maxLength = ArgumentsUtil.intValue(arguments, "maxLength", 960);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        NDArray result = list.singletonOrThrow();
        result = result.squeeze().mul(255f).toType(DataType.UINT8, true).neq(0);
        boolean[] flattened = result.toBooleanArray();
        Shape shape = result.getShape();
        int w = (int) shape.get(0);
        int h = (int) shape.get(1);
        boolean[][] grid = new boolean[w][h];
        IntStream.range(0, flattened.length)
                .parallel()
                .forEach(i -> grid[i / h][i % h] = flattened[i]);
        List<BoundingBox> boxes = new BoundFinder(grid).getBoxes();
        List<String> names = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        int boxSize = boxes.size();
        for (int i = 0; i < boxSize; i++) {
            names.add("word");
            probs.add(1.0);
        }
        return new DetectedObjects(names, probs, boxes);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray img = input.toNDArray(ctx.getNDManager());
        int h = input.getHeight();
        int w = input.getWidth();
        int[] hw = scale(h, w, maxLength);

        img = NDImageUtils.resize(img, hw[1], hw[0]);
        img = NDImageUtils.toTensor(img);
        img =
                NDImageUtils.normalize(
                        img,
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f});
        img = img.expandDims(0);
        return new NDList(img);
    }

    private int[] scale(int h, int w, int max) {
        int localMax = Math.max(h, w);
        float scale = 1.0f;
        if (max < localMax) {
            scale = max * 1.0f / localMax;
        }
        // paddle model only take 32-based size
        return resize32(h * scale, w * scale);
    }

    private int[] resize32(double h, double w) {
        double min = Math.min(h, w);
        if (min < 32) {
            h = 32.0 / min * h;
            w = 32.0 / min * w;
        }
        int h32 = (int) h / 32;
        int w32 = (int) w / 32;
        return new int[] {h32 * 32, w32 * 32};
    }
}
