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
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.RandomUtils;

import java.nio.ByteBuffer;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries at the detailed pixel level.
 */
public class SemanticSegmentationTranslator extends BaseImageTranslator<Image> {

    private final int shortEdge;
    private final int maxEdge;

    private static final int CHANNEL = 3;
    private static final int CLASSNUM = 21;

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
        float[] scores = list.get(1).toFloatArray();
        Shape shape = list.get(1).getShape();
        int width = (int) shape.get(2);
        int height = (int) shape.get(1);

        // build image array
        try (NDManager manager = NDManager.newBaseManager()) {
            int imageSize = width * height;
            ByteBuffer bb = manager.allocateDirect(CHANNEL * imageSize);
            int r = 0; // adjustment for red pixel
            int g = 1; // adjustment for green pixel
            int b = 2; // adjustment for blue pixel
            byte[][] colors = new byte[CLASSNUM][3];
            for (int i = 0; i < CLASSNUM; i++) {
                byte red = (byte) RandomUtils.nextInt(256);
                byte green = (byte) RandomUtils.nextInt(256);
                byte blue = (byte) RandomUtils.nextInt(256);
                colors[i][r] = red;
                colors[i][g] = green;
                colors[i][b] = blue;
            }

            // change color of pixels in image array where objects have been detected
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index = h * width + w;
                    int maxi = 0;
                    double maxnum = -Double.MAX_VALUE;
                    for (int i = 0; i < CLASSNUM; i++) {
                        // get score for each i at the h,w pixel of the image
                        float score = scores[i * (imageSize) + index];
                        if (score > maxnum) {
                            maxnum = score;
                            maxi = i;
                        }
                    }
                    if (maxi > 0) {
                        bb.put(colors[maxi][r]);
                        bb.put(colors[maxi][g]);
                        bb.put(colors[maxi][b]);
                    } else {
                        bb.position(bb.position() + 3);
                    }
                }
            }
            bb.rewind();
            int originW = (int) ctx.getAttachment("originalWidth");
            int originH = (int) ctx.getAttachment("originalHeight");
            NDArray fullImage =
                    manager.create(bb, new Shape(height, width, CHANNEL), DataType.UINT8);
            NDArray resized = NDImageUtils.resize(fullImage, originW, originH);

            return ImageFactory.getInstance().fromNDArray(resized);
        }
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
            int rescaledHeight = Math.round(height * scale);
            int rescaledWidth = Math.round(width * scale);

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
