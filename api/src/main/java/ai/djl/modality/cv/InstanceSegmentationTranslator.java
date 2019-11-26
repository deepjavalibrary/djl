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

import ai.djl.Model;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries at the detailed pixel level.
 */
public class InstanceSegmentationTranslator extends ImageTranslator<DetectedObjects>
        implements Transform {

    private String synsetArtifactName;
    private float threshold;
    private int shortEdge;
    private int maxEdge;

    private int rescaledWidth;
    private int rescaledHeight;

    /**
     * Creates the Instance Segmentation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public InstanceSegmentationTranslator(Builder builder) {
        super(builder);
        synsetArtifactName = builder.synsetArtifactName;
        this.threshold = builder.threshold;
        this.shortEdge = builder.shortEdge;
        this.maxEdge = builder.maxEdge;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        return resizeShort(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage image) {
        Pipeline pipeline = getPipeline();
        pipeline.insert(0, null, this);
        ctx.setAttachment("originalHeight", image.getHeight());
        ctx.setAttachment("originalWidth", image.getWidth());
        return super.processInput(ctx, image);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();
        List<String> classes = model.getArtifact(synsetArtifactName, Utils::readLines);

        float[] ids = list.get(0).toFloatArray();
        float[] scores = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);
        NDArray masks = list.get(3);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < ids.length; ++i) {
            int classId = (int) ids[i];
            double probability = scores[i];
            if (classId >= 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                double x = box[0] / rescaledWidth;
                double y = box[1] / rescaledHeight;
                double w = box[2] / rescaledWidth - x;
                double h = box[3] / rescaledHeight - y;

                int maskW = (int) (w * (int) ctx.getAttachment("originalWidth"));
                int maskH = (int) (h * (int) ctx.getAttachment("originalHeight"));

                // Reshape mask to actual image bounding box shape.
                NDArray array = masks.get(i);
                Shape maskShape = array.getShape();
                array = array.reshape(maskShape.addAll(new Shape(1)));
                NDArray maskArray = NDImageUtils.resize(array, maskW, maskH).transpose();
                float[] flattened = maskArray.toFloatArray();
                float[][] maskFloat = new float[maskW][maskH];
                for (int j = 0; j < maskW; j++) {
                    System.arraycopy(flattened, j * maskH, maskFloat[j], 0, maskH);
                }
                Mask mask = new Mask(x, y, w, h, maskFloat);

                retNames.add(className);
                retProbs.add(probability);
                retBB.add(mask);
            }
        }
        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /**
     * Resizes the image based on the shorter edge or maximum edge length.
     *
     * @param image the input image
     * @return the resized image
     */
    private NDArray resizeShort(NDArray image) {
        Shape shape = image.getShape();
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

        return NDImageUtils.resize(image, rescaledWidth, rescaledHeight);
    }

    /** The builder for Instance Segmentation translator. */
    public static class Builder extends BaseBuilder<Builder> {

        private String synsetArtifactName;
        private float threshold = 0.3f;
        private int shortEdge = 600;
        private int maxEdge = 1000;

        /**
         * Sets the name for the synset.
         *
         * <p>synset is used to convert the prediction classes to their actual names.
         *
         * @param synsetArtifactName the name of synset
         * @return the builder
         */
        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for prediction accuracy
         * @return the builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Sets the shorter edge length of the rescaled image.
         *
         * @param shortEdge the length of the short edge
         * @return the builder
         */
        public Builder optShortEdge(int shortEdge) {
            this.shortEdge = shortEdge;
            return this;
        }

        /**
         * Sets the maximum edge length of the rescaled image.
         *
         * @param maxEdge the length of the longest edge
         * @return the builder
         */
        public Builder optMaxEdge(int maxEdge) {
            this.maxEdge = maxEdge;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public InstanceSegmentationTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new InstanceSegmentationTranslator(this);
        }
    }
}
