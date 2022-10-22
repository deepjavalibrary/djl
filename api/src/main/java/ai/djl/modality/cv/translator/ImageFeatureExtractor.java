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
package ai.djl.modality.cv.translator;

import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;

import java.util.Map;

/**
 * A generic {@link ai.djl.translate.Translator} for Image Classification feature extraction tasks.
 */
public class ImageFeatureExtractor extends BaseImageTranslator<byte[]> {

    /**
     * Constructs an Image Classification using {@link Builder}.
     *
     * @param builder the data to build with
     */
    ImageFeatureExtractor(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public byte[] processOutput(TranslatorContext ctx, NDList list) {
        return list.get(0).toByteArray();
    }

    /**
     * Creates a builder to build a {@code ImageFeatureExtractor}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code ImageFeatureExtractor} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();
        builder.configPreProcess(arguments);
        return builder;
    }

    /** A Builder to construct a {@code ImageFeatureExtractor}. */
    public static class Builder extends BaseBuilder<Builder> {

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the {@link ImageFeatureExtractor} with the provided data.
         *
         * @return an {@link ImageFeatureExtractor}
         */
        public ImageFeatureExtractor build() {
            validate();
            return new ImageFeatureExtractor(this);
        }
    }
}
