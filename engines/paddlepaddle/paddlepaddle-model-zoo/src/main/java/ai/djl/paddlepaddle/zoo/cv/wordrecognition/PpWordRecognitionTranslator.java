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

package ai.djl.paddlepaddle.zoo.cv.wordrecognition;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * A {@link PpWordRecognitionTranslator} that preprocess {@link Image} post-process the {@link
 * NDArray} into text.
 */
public class PpWordRecognitionTranslator implements NoBatchifyTranslator<Image, String> {

    private List<String> table;

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        try (InputStream is =
                ctx.getModel().getArtifact("rec_crnn/ppocr_keys_v1.txt").openStream()) {
            table = Utils.readLines(is, true);
            table.add(0, "blank");
            table.add("");
        }
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        StringBuilder sb = new StringBuilder();
        NDArray tokens = list.singletonOrThrow();
        long[] indices = tokens.get(0).argMax(1).toLongArray();
        int lastIdx = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] > 0 && !(i > 0 && indices[i] == lastIdx)) {
                sb.append(table.get((int) indices[i]));
            }
        }
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray img = input.toNDArray(ctx.getNDManager());
        int[] hw = resize32(input.getWidth());
        img = NDImageUtils.resize(img, hw[1], hw[0]);
        img = NDImageUtils.toTensor(img).sub(0.5f).div(0.5f);
        img = img.expandDims(0);
        return new NDList(img);
    }

    private int[] resize32(double w) {
        // Paddle doesn't rely on aspect ratio
        int width = ((int) Math.max(32, w)) / 32 * 32;
        return new int[] {32, width};
    }
}
