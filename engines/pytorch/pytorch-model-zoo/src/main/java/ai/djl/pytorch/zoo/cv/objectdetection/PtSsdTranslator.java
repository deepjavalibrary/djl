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

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.ObjectDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link PtSsdTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries. Reference implementation: <a
 * href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD">SSD</a>.
 */
public class PtSsdTranslator extends ObjectDetectionTranslator {

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
    protected PtSsdTranslator(Builder builder) {
        super(builder);
        this.figSize = builder.figSize;
        this.featSize = builder.featSize;
        this.steps = builder.steps;
        this.scale = builder.scale;
        this.aspectRatio = builder.aspectRatio;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        super.prepare(ctx);
        NDManager manager = ctx.getPredictorManager();
        boxRecover = boxRecover(manager, figSize, featSize, steps, scale, aspectRatio);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        double scaleXY = 0.1;
        double scaleWH = 0.2;

        // kill the 1st prediction as not needed
        NDArray prob = list.get(1).swapAxes(0, 1).softmax(1).get(":, 1:");
        prob =
                NDArrays.stack(
                        new NDList(
                                prob.argMax(1).toType(DataType.FLOAT32, false),
                                prob.max(new int[] {1})));
        NDArray boundingBoxes = list.get(0).swapAxes(0, 1);
        NDArray bbWH = boundingBoxes.get(":, 2:").mul(scaleWH).exp().mul(boxRecover.get(":, 2:"));
        NDArray bbXY =
                boundingBoxes
                        .get(":, :2")
                        .mul(scaleXY)
                        .mul(boxRecover.get(":, 2:"))
                        .add(boxRecover.get(":, :2"))
                        .sub(bbWH.mul(0.5f));
        boundingBoxes = NDArrays.concat(new NDList(bbXY, bbWH), 1);
        // filter the result below the threshold
        NDArray cutOff = prob.get(1).gte(threshold);
        boundingBoxes = boundingBoxes.transpose().booleanMask(cutOff, 1).transpose();
        prob = prob.booleanMask(cutOff, 1);
        // start categorical filtering
        long[] order = prob.get(1).argSort().toLongArray();
        double desiredIoU = 0.45;
        prob = prob.transpose();
        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        Map<Integer, List<BoundingBox>> recorder = new ConcurrentHashMap<>();

        for (int i = order.length - 1; i >= 0; i--) {
            long currMaxLoc = order[i];
            float[] classProb = prob.get(currMaxLoc).toFloatArray();
            int classId = (int) classProb[0];
            double probability = classProb[1];
            double[] boxArr = boundingBoxes.get(currMaxLoc).toDoubleArray();
            Rectangle rect = new Rectangle(boxArr[0], boxArr[1], boxArr[2], boxArr[3]);
            List<BoundingBox> boxes = recorder.getOrDefault(classId, new ArrayList<>());
            boolean belowIoU = true;
            for (BoundingBox box : boxes) {
                if (box.getIoU(rect) > desiredIoU) {
                    belowIoU = false;
                    break;
                }
            }
            if (belowIoU) {
                boxes.add(rect);
                recorder.put(classId, boxes);
                String className = classes.get(classId);
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

    /**
     * Creates a builder to build a {@code PtSSDTranslatorBuilder} with specified arguments.
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
        @SuppressWarnings("unchecked")
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);

            threshold = ArgumentsUtil.floatValue(arguments, "threshold", 0.4f);
            figSize = ArgumentsUtil.intValue(arguments, "size", 300);
            List<Double> list = (List<Double>) arguments.get("featSize");
            if (list == null) {
                featSize = new int[] {38, 19, 10, 5, 3, 1};
            } else {
                featSize = list.stream().mapToInt(Double::intValue).toArray();
            }
            list = (List<Double>) arguments.get("steps");
            if (list == null) {
                steps = new int[] {8, 16, 32, 64, 100, 300};
            } else {
                steps = list.stream().mapToInt(Double::intValue).toArray();
            }

            list = (List<Double>) arguments.get("scale");
            if (list == null) {
                scale = new int[] {21, 45, 99, 153, 207, 261, 315};
            } else {
                scale = list.stream().mapToInt(Double::intValue).toArray();
            }

            List<List<Double>> ratio = (List<List<Double>>) arguments.get("aspectRatios");
            if (ratio == null) {
                aspectRatio = new int[][] {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};
            } else {
                aspectRatio = new int[ratio.size()][];
                for (int i = 0; i < aspectRatio.length; ++i) {
                    aspectRatio[i] = ratio.get(i).stream().mapToInt(Double::intValue).toArray();
                }
            }
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public PtSsdTranslator build() {
            validate();
            return new PtSsdTranslator(this);
        }
    }
}
