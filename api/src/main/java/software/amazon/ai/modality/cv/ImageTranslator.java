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
package software.amazon.ai.modality.cv;

import java.awt.image.BufferedImage;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils.Flag;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;

/**
 * Built-in {@code Translator} that provides default image pre-processing.
 *
 * @param <T> output object type
 */
public abstract class ImageTranslator<T> implements Translator<BufferedImage, T> {

    protected Flag flag;
    private boolean centerCrop;
    private int[] centerCropSize;
    private int[] resize;
    protected float[] normalizeMeans;
    protected float[] normalizeStd;

    public ImageTranslator(BaseBuilder<?> builder) {
        flag = builder.flag;
        centerCrop = builder.centerCrop;
        centerCropSize = builder.centerCropSize;
        resize = builder.resize;
        normalizeMeans = builder.normalizeMeans;
        normalizeStd = builder.normalizeStd;
    }

    /**
     * Processes the {@code BufferedImage} input and converts it to NDList.
     *
     * @param ctx toolkit that helps create input NDArray
     * @param input {@code BufferedImage} input
     * @return {@link NDList}
     */
    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        if (centerCrop) {
            if (centerCropSize != null) {
                input = BufferedImageUtils.centerCrop(input, centerCropSize[1], centerCropSize[0]);
            } else {
                input = BufferedImageUtils.centerCrop(input);
            }
        }
        if (resize != null) {
            input = BufferedImageUtils.resize(input, resize[1], resize[0]);
        }
        NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), input, flag);
        return new NDList(normalize(array.divi(255)));
    }

    /**
     * Normalizes a pre-processed {@link NDArray}.
     *
     * <p>It's expected that the developer overrides this method to provide customized
     * normalization.
     *
     * @param array pre-processed {@link NDArray}
     * @return normalized NDArray
     */
    protected NDArray normalize(NDArray array) {
        if (normalizeMeans != null && normalizeStd != null) {
            array = NDImageUtils.normalize(array, normalizeMeans, normalizeStd);
        }
        return array;
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Flag flag = Flag.COLOR;
        private boolean centerCrop;
        private int[] centerCropSize;
        private int[] resize;
        protected float[] normalizeMeans;
        protected float[] normalizeStd;

        public T optFlag(Flag flag) {
            this.flag = flag;
            return self();
        }

        public T optCenterCrop() {
            centerCrop = true;
            centerCropSize = null;
            return self();
        }

        public T optCenterCrop(int height, int width) {
            centerCropSize = new int[] {height, width};
            return self();
        }

        public T optResize(int height, int width) {
            resize = new int[] {height, width};
            return self();
        }

        public T optNormalize(float[] normalizeMeans, float[] normalizeStd) {
            this.normalizeMeans = normalizeMeans;
            this.normalizeStd = normalizeStd;
            return self();
        }

        protected abstract T self();
    }
}
