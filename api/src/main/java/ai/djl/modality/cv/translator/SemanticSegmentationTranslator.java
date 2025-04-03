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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.transform.ResizeShort;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * A {@link Translator} that post-process the {@link Image} into {@link CategoryMask} with output
 * mask representing the class that each pixel in the original image belong to.
 */
public class SemanticSegmentationTranslator extends BaseImageTranslator<CategoryMask> {

    private SynsetLoader synsetLoader;

    private List<String> classes;

    /**
     * Creates the Semantic Segmentation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SemanticSegmentationTranslator(Builder builder) {
        super(builder);
        this.synsetLoader = builder.synsetLoader;
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
    public CategoryMask processOutput(TranslatorContext ctx, NDList list) {
        // scores contains the probabilities of each pixel being a certain object
        float[] scores = list.get(1).toFloatArray();
        Shape shape = list.get(1).getShape();
        int width = (int) shape.get(2);
        int height = (int) shape.get(1);
        int[][] mask = new int[height][width];

        int imageSize = width * height;

        // Build mask array
        int numOfClasses = classes.size();
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int index = h * width + w;
                int maxi = 0;
                double maxnum = -Double.MAX_VALUE;
                for (int i = 0; i < numOfClasses; ++i) {
                    // get score for each i at the h,w pixel of the image
                    float score = scores[i * imageSize + index];
                    if (score > maxnum) {
                        maxnum = score;
                        maxi = i;
                    }
                }
                mask[h][w] = maxi;
            }
        }
        return new CategoryMask(classes, mask);
    }

    /**
     * Creates a builder to build a {@code SemanticSegmentationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code SemanticSegmentationTranslator} with specified arguments.
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

    /** The builder for Semantic Segmentation translator. */
    public static class Builder extends ClassificationBuilder<Builder> {
        int shortEdge = 600;
        int maxEdge = 1000;

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            shortEdge = ArgumentsUtil.intValue(arguments, "shortEdge", 600);
            maxEdge = ArgumentsUtil.intValue(arguments, "maxEdge", 1000);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SemanticSegmentationTranslator build() {
            validate();
            ResizeShort resize = new ResizeShort(shortEdge, maxEdge, Image.Interpolation.BILINEAR);
            pipeline.insert(0, null, resize);

            return new SemanticSegmentationTranslator(this);
        }
    }
}
