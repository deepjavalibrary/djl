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
package ai.djl.modality.cv;

import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.awt.image.BufferedImage;

/**
 * Built-in {@code Translator} that provides default image pre-processing.
 *
 * @param <T> the output object type
 */
public abstract class ImageTranslator<T> implements Translator<BufferedImage, T> {

    private NDImageUtils.Flag flag;
    private Pipeline pipeline;

    public ImageTranslator(BaseBuilder<?> builder) {
        flag = builder.flag;
        pipeline = builder.pipeline;
    }

    /** {@inheritDoc} */
    @Override
    public Pipeline getPipeline() {
        return pipeline;
    }

    /**
     * Processes the {@code BufferedImage} input and converts it to NDList.
     *
     * @param ctx the toolkit that helps create the input NDArray
     * @param input the {@code BufferedImage} input
     * @return a {@link NDList}
     */
    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), input, flag);
        return pipeline.transform(new NDList(array));
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected NDImageUtils.Flag flag = NDImageUtils.Flag.COLOR;
        protected Pipeline pipeline;

        public T optFlag(NDImageUtils.Flag flag) {
            this.flag = flag;
            return self();
        }

        public T setPipeline(Pipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        protected abstract T self();
    }
}
