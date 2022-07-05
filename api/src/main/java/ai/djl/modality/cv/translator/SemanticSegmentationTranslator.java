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

import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslatorContext;

import java.nio.ByteBuffer;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries at the detailed pixel level.
 */
public class SemanticSegmentationTranslator extends BaseImageTranslator<Image> {
    private final int shortEdge;
    private final int maxEdge;

    private int rescaledWidth;
    private int rescaledHeight;

    private static final int CHANNEL = 3;
    private static final int CLASSNUM = 21;
    private static final int BIKE = 2;
    private static final int CAR = 7;
    private static final int DOG = 8;
    private static final int CAT = 12;
    private static final int PERSON = 15;

    // sheep is also identified with id 13 as well, this is taken into account when coloring pixels
    private static final int SHEEP = 17; // 13

    /**
     * Creates the Semantic Segmentation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SemanticSegmentationTranslator(Builder builder) {
        super(builder);
        this.shortEdge = builder.shortEdge;
        this.maxEdge = builder.maxEdge;

        pipeline.insert(0, null, new ResizeShort());
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image image) {
        ctx.setAttachment("originalHeight", image.getHeight());
        ctx.setAttachment("originalWidth", image.getWidth());
        return super.processInput(ctx, image);
    }

    /** {@inheritDoc} */
    @Override
    public Image processOutput(TranslatorContext ctx, NDList list) {
        // scores contains the probabilities of each pixel being a certain object
        final float[] scores = list.get(1).toFloatArray();

        // get dimensions of image
        final int width = (int) ctx.getAttachment("originalWidth");
        final int height = (int) ctx.getAttachment("originalHeight");

        // build image array
        NDManager manager = NDManager.newBaseManager();
        ByteBuffer bb = manager.allocateDirect(CHANNEL * height * width);
        NDArray intRet = manager.create(bb, new Shape(CHANNEL, height, width), DataType.UINT8);

        // change color of pixels in image array where objects have been detected
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                int maxi = 0;
                double maxnum = -Double.MAX_VALUE;
                for (int i = 0; i < CLASSNUM; i++) {

                    // get score for each i at the k,j pixel of the image
                    float score = scores[i * (width * height) + j * width + k];
                    if (score > maxnum) {
                        maxnum = score;
                        maxi = i;
                    }
                }

                // color pixel if object was found, otherwise leave as is (black)
                if (maxi == PERSON || maxi == BIKE) {
                    NDIndex index = new NDIndex(0, j, k);
                    intRet.set(index, 0xFF00FF);
                } else if (maxi == CAT || maxi == SHEEP || maxi == 13) {
                    NDIndex index = new NDIndex(1, j, k);
                    intRet.set(index, 0xFF00FF);
                } else if (maxi == CAR || maxi == DOG) {
                    NDIndex index = new NDIndex(2, j, k);
                    intRet.set(index, 0xFF00FF);
                }
            }
        }
        return BufferedImageFactory.getInstance().fromNDArray(intRet);
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

    /** Resizes the image based on the shorter edge or maximum edge length. */
    private class ResizeShort implements Transform {
        /** {@inheritDoc} */
        @Override
        public NDArray transform(NDArray array) {
            Shape shape = array.getShape();
            int width = (int) shape.get(1);
            int height = (int) shape.get(0);
            int min = Math.min(width, height);
            int max = Math.max(width, height);
            float scale = shortEdge / (float) min;
            if (Math.round(scale * max) > maxEdge) {
                scale = maxEdge / (float) max;
            }
            rescaledHeight = Math.round(height * scale);
            rescaledWidth = Math.round(width * scale);

            return NDImageUtils.resize(array, rescaledWidth, rescaledHeight);
        }
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
            return new SemanticSegmentationTranslator(this);
        }
    }
}
