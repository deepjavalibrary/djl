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
package ai.djl.paddlepaddle.zoo;

import ai.djl.Model;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.ObjectDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.Map;

/**
 * A {@link PpFaceDetectionTranslator} that post-process the {@link NDArray} into {@link
 * DetectedObjects} with boundaries.
 */
public class PpFaceDetectionTranslator extends ObjectDetectionTranslator {

    /**
     * Creates the FaceDetection translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    protected PpFaceDetectionTranslator(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        super.prepare(manager, model);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        // TODO: add implementation here
        return null;
    }

    /**
     * Creates a builder to build a {@code PpFaceDetectionTranslatorBuilder}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code PpFaceDetectionTranslatorBuilder} with specified
     * arguments.
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

    /** The builder for SSD translator. */
    public static class Builder extends ObjectDetectionBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPreProcess(Map<String, ?> arguments) {
            super.configPreProcess(arguments);
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public PpFaceDetectionTranslator build() {
            validate();
            return new PpFaceDetectionTranslator(this);
        }
    }
}
