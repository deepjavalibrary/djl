/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class ImageEncoder implements NoBatchifyTranslator<Image, NDArray> {

    private int height;
    private int width;

    public ImageEncoder(int height, int width) {
        this.height = height;
        this.width = width;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray result = list.singletonOrThrow();
        result = result.mul(0.18215f);
        result.detach();
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        // model take 32-based size
        int[] size = resize32(height, width);

        array = NDImageUtils.resize(array, size[1], size[0]);
        array = array.transpose(2, 0, 1).div(255f); // HWC -> CHW RGB
        array = array.mul(2).sub(1);
        array = array.expandDims(0);
        return new NDList(array);
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
