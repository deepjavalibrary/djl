/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A translator for Yolov8 pose estimation models. */
public class YoloPoseTranslator extends BaseImageTranslator<Joints[]> {

    private static final int MAX_DETECTION = 300;

    private float threshold;
    private float nmsThreshold;

    /**
     * Creates the Pose Estimation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public YoloPoseTranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.nmsThreshold = builder.nmsThreshold;
    }

    /** {@inheritDoc} */
    @Override
    public Joints[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray pred = list.singletonOrThrow();
        NDArray candidates = pred.get(4).gt(threshold);
        pred = pred.transpose();
        NDArray sub = pred.get("..., :4");
        sub = YoloTranslator.xywh2xyxy(sub);
        pred = sub.concat(pred.get("..., 4:"), -1);
        pred = pred.get(candidates);

        NDList split = pred.split(new long[] {4, 5}, 1);
        NDArray box = split.get(0);

        int numBox = Math.toIntExact(box.getShape().get(0));

        float[] buf = box.toFloatArray();
        float[] confidences = split.get(1).toFloatArray();
        float[] mask = split.get(2).toFloatArray();

        List<Rectangle> boxes = new ArrayList<>(numBox);
        List<Double> scores = new ArrayList<>(numBox);

        for (int i = 0; i < numBox; ++i) {
            float xPos = buf[i * 4];
            float yPos = buf[i * 4 + 1];
            float w = buf[i * 4 + 2] - xPos;
            float h = buf[i * 4 + 3] - yPos;
            Rectangle rect = new Rectangle(xPos, yPos, w, h);
            boxes.add(rect);
            scores.add((double) confidences[i]);
        }
        List<Integer> nms = Rectangle.nms(boxes, scores, nmsThreshold);
        if (nms.size() > MAX_DETECTION) {
            nms = nms.subList(0, MAX_DETECTION);
        }
        Joints[] ret = new Joints[nms.size()];
        for (int i = 0; i < ret.length; ++i) {
            List<Joint> joints = new ArrayList<>();
            ret[i] = new Joints(joints);

            int index = nms.get(i);
            int pos = index * 51;
            for (int j = 0; j < 17; ++j) {
                joints.add(
                        new Joints.Joint(
                                mask[pos + j * 3] / width,
                                mask[pos + j * 3 + 1] / height,
                                mask[pos + j * 3 + 2]));
            }
        }
        return ret;
    }

    /**
     * Creates a builder to build a {@code YoloPoseTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code YoloPoseTranslator} with specified arguments.
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

    /** The builder for Pose Estimation translator. */
    public static class Builder extends BaseBuilder<Builder> {

        float threshold = 0.25f;
        float nmsThreshold = 0.7f;

        Builder() {}

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
            return self();
        }

        /**
         * Sets the NMS threshold.
         *
         * @param nmsThreshold the NMS threshold
         * @return this builder
         */
        public Builder optNmsThreshold(float nmsThreshold) {
            this.nmsThreshold = nmsThreshold;
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
            optThreshold(ArgumentsUtil.floatValue(arguments, "threshold", threshold));
            optNmsThreshold(ArgumentsUtil.floatValue(arguments, "nmsThreshold", nmsThreshold));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public YoloPoseTranslator build() {
            validate();
            return new YoloPoseTranslator(this);
        }
    }
}
