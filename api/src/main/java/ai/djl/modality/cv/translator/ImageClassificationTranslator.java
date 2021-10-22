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

package ai.djl.modality.cv.translator;

import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/** A generic {@link ai.djl.translate.Translator} for Image Classification tasks. */
public class ImageClassificationTranslator extends BaseImageTranslator<Classifications> {

    private SynsetLoader synsetLoader;
    private boolean applySoftmax;

    private List<String> classes;

    /**
     * Constructs an Image Classification using {@link Builder}.
     *
     * @param builder the data to build with
     */
    public ImageClassificationTranslator(Builder builder) {
        super(builder);
        this.synsetLoader = builder.synsetLoader;
        this.applySoftmax = builder.applySoftmax;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        if (classes == null) {
            classes = synsetLoader.load(ctx.getModel());
        }
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilitiesNd = list.singletonOrThrow();
        if (applySoftmax) {
            probabilitiesNd = probabilitiesNd.softmax(0);
        }
        return new Classifications(classes, probabilitiesNd);
    }

    /**
     * Creates a builder to build a {@code ImageClassificationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code ImageClassificationTranslator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);
        return builder;
    }

    /** A Builder to construct a {@code ImageClassificationTranslator}. */
    public static class Builder extends ClassificationBuilder<Builder> {

        private boolean applySoftmax;

        Builder() {}

        /**
         * Sets whether to apply softmax when processing output. Some models already include softmax
         * in the last layer, so don't apply softmax when processing model output.
         *
         * @param applySoftmax boolean whether to apply softmax
         * @return the builder
         */
        public Builder optApplySoftmax(boolean applySoftmax) {
            this.applySoftmax = applySoftmax;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            applySoftmax = ArgumentsUtil.booleanValue(arguments, "applySoftmax");
        }

        /**
         * Builds the {@link ImageClassificationTranslator} with the provided data.
         *
         * @return an {@link ImageClassificationTranslator}
         */
        public ImageClassificationTranslator build() {
            validate();
            return new ImageClassificationTranslator(this);
        }
    }
}
