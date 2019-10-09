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
package org.apache.mxnet.zoo.cv.segmentation;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.Model;
import software.amazon.ai.modality.cv.BoundingBox;
import software.amazon.ai.modality.cv.DetectedObjects;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.Mask;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Utils;

public class InstanceSegementationTranslator extends ImageTranslator<DetectedObjects> {

    private String synsetArtifactName;
    private float threshold;
    private int shortEdge;
    private int maxEdge;

    private int rescaledWidth;
    private int rescaledHeight;

    public InstanceSegementationTranslator(Builder builder) {
        super(builder);
        synsetArtifactName = builder.synsetArtifactName;
        this.threshold = builder.threshold;
        this.shortEdge = builder.shortEdge;
        this.maxEdge = builder.maxEdge;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage image) {
        image = resizeShort(image);
        rescaledWidth = image.getWidth();
        rescaledHeight = image.getHeight();
        return super.processInput(ctx, image);
    }

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
                float[] box = boundingBoxes.get(0, i).toFloatArray();
                double x = box[0] / rescaledWidth;
                double y = box[1] / rescaledHeight;
                double w = box[2] / rescaledWidth - x;
                double h = box[3] / rescaledHeight - y;

                Shape maskShape = masks.get(0, i).getShape();
                float[][] maskVal = new float[(int) maskShape.get(0)][(int) maskShape.get(1)];
                float[] flattened = masks.get(0, i).toFloatArray();

                for (int j = 0; j < flattened.length; j++) {
                    maskVal[j / maskVal.length][j % maskVal.length] = flattened[j];
                }

                Mask mask = new Mask(x, y, w, h, maskVal);

                retNames.add(className);
                retProbs.add(probability);
                retBB.add(mask);
            }
        }
        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /**
     * resize the image based on the shorter edge or maximum edge length.
     *
     * @param img the input image
     * @return resized image
     */
    private BufferedImage resizeShort(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int min = Math.min(width, height);
        int max = Math.max(width, height);
        float scale = shortEdge / (float) min;
        if (Math.round(scale * max) > maxEdge) {
            scale = maxEdge / (float) max;
        }
        width = Math.round(width * scale);
        height = Math.round(height * scale);

        return BufferedImageUtils.resize(img, width, height);
    }

    public static class Builder extends BaseBuilder<Builder> {

        private String synsetArtifactName;
        private float threshold = 0.3f;
        private int shortEdge = 600;
        private int maxEdge = 1000;

        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder optShortEdge(int shortEdge) {
            this.shortEdge = shortEdge;
            return this;
        }

        public Builder optMaxEdge(int maxEdge) {
            this.maxEdge = maxEdge;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public InstanceSegementationTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new InstanceSegementationTranslator(this);
        }
    }
}
