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
package ai.djl.pytorch.zoo.cv.objectdetection;

import ai.djl.Model;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link PtSSDTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries. Reference implementation: <a
 * href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD">SSD</a>.
 */
public class PtSSDTranslator extends SingleShotDetectionTranslator {

    private NDArray boxRecover;
    private int figSize;
    private int[] featSize;
    private int[] steps;
    private int[] scale;
    private int[][] aspectRatio;

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public PtSSDTranslator(Builder builder) {
        super(builder);
        this.figSize = builder.figSize;
        this.featSize = builder.featSize;
        this.steps = builder.steps;
        this.scale = builder.scale;
        this.aspectRatio = builder.aspectRatio;
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();
        double scaleXY = 0.1;
        double scaleWH = 0.2;
        if (classes == null) {
            classes = model.getArtifact(synsetArtifactName, Utils::readLines);
        }
        if (boxRecover == null) {
            boxRecover =
                    boxRecover(model.getNDManager(), figSize, featSize, steps, scale, aspectRatio);
        }

        // kill the 1st prediction as not needed
        NDArray prob = list.get(1).swapAxes(0, 1).softmax(1).get(":, 1:");
        NDArray boundingBoxes = list.get(0).swapAxes(0, 1);
        NDArray bbXY =
                boundingBoxes
                        .get(":, :2")
                        .mul(scaleXY)
                        .mul(boxRecover.get(":, 2:"))
                        .add(boxRecover.get(":, :2"));
        NDArray bbWH = boundingBoxes.get(":, 2:").mul(scaleWH).exp().mul(boxRecover.get(":, 2:"));
        boundingBoxes = NDArrays.concat(new NDList(bbXY, bbWH), 1);
        long[] classIds = prob.argMax(1).toLongArray();
        float[] probabilities = prob.max(new int[] {1}).toFloatArray();

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < classIds.length; ++i) {
            long classId = classIds[i];
            double probability = probabilities[i];
            // classId starts from 0, -1 means background
            if (classId >= 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get((int) classId);
                double[] box = boundingBoxes.get(i).toDoubleArray();

                Rectangle rect = new Rectangle(box[0], box[1], box[2], box[3]);
                retNames.add(className);
                retProbs.add(probability);
                retBB.add(rect);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    NDArray boxRecover(
            NDManager manager,
            int figSize,
            int[] featSize,
            int[] steps,
            int[] scale,
            int[][] aspectRatio) {
        double[] fk =
                manager.create(steps)
                        .toType(DataType.FLOAT64, true)
                        .getNDArrayInternal()
                        .rdiv((double) figSize)
                        .toDoubleArray();

        List<double[]> defaultBoxes = new ArrayList<>();

        for (int idx = 0; idx < featSize.length; idx++) {
            double sk1 = scale[idx] * 1.0 / figSize;
            double sk2 = scale[idx + 1] * 1.0 / figSize;
            double sk3 = Math.sqrt(sk1 * sk2);
            List<double[]> array = new ArrayList<>();
            array.add(new double[] {sk1, sk1});
            array.add(new double[] {sk3, sk3});

            for (int alpha : aspectRatio[idx]) {
                double w = sk1 * Math.sqrt(alpha);
                double h = sk1 / Math.sqrt(alpha);
                array.add(new double[] {w, h});
                array.add(new double[] {h, w});
            }
            for (double[] size : array) {
                for (int i = 0; i < featSize[idx]; i++) {
                    for (int j = 0; j < featSize[idx]; j++) {
                        double cx = (j + 0.5) / fk[idx];
                        double cy = (i + 0.5) / fk[idx];
                        defaultBoxes.add(new double[] {cx, cy, size[0], size[1]});
                    }
                }
            }
        }
        double[][] boxes = new double[defaultBoxes.size()][defaultBoxes.get(0).length];
        for (int i = 0; i < defaultBoxes.size(); i++) {
            boxes[i] = defaultBoxes.get(i);
        }
        return manager.create(boxes).clip(0.0, 1.0);
    }

    /**
     * Creates a builder to build a {@code PtSSDTranslatorBuilder}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for SSD translator. */
    public static class Builder extends SingleShotDetectionTranslator.Builder {

        private int figSize;
        private int[] featSize;
        private int[] steps;
        private int[] scale;
        private int[][] aspectRatio;

        /**
         * Set the box parameter to reconstruct the anchor box.
         *
         * @param figSize image size
         * @param featSize feature size
         * @param steps steps to create boxes
         * @param scale scale between different level of generated boxes
         * @param aspectRatio parameter go along with scale
         * @return this builder
         */
        public Builder setBoxes(
                int figSize, int[] featSize, int[] steps, int[] scale, int[][] aspectRatio) {
            this.figSize = figSize;
            this.featSize = featSize;
            this.steps = steps;
            this.scale = scale;
            this.aspectRatio = aspectRatio;
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        @Override
        public PtSSDTranslator build() {
            if (getSynsetArtifactName() == null && getClasses() == null) {
                throw new IllegalArgumentException(
                        "You must specify a synset artifact name or classes");
            } else if (getSynsetArtifactName() != null && getClasses() != null) {
                throw new IllegalArgumentException(
                        "You can only specify one of: synset artifact name or classes");
            }
            return new PtSSDTranslator(this);
        }
    }
}
