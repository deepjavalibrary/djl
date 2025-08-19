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

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A translator for Yolov8 instance segmentation models. */
public class YoloSegmentationTranslator extends YoloV5Translator {

    private static final int[] AXIS_0 = {0};
    private static final int[] AXIS_1 = {1};

    private float threshold;
    private float nmsThreshold;

    /**
     * Creates the instance segmentation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public YoloSegmentationTranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.nmsThreshold = builder.nmsThreshold;
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        NDArray pred = list.get(0);
        NDArray protos = list.get(1);
        int maskIndex = classes.size() + 4;
        NDArray candidates = pred.get("4:" + maskIndex).max(AXIS_0).gt(threshold);
        pred = pred.transpose();
        NDArray sub = pred.get("..., :4");
        sub = YoloTranslator.xywh2xyxy(sub);
        pred = sub.concat(pred.get("..., 4:"), -1);
        pred = pred.get(candidates);

        NDList split = pred.split(new long[] {4, maskIndex}, 1);
        NDArray box = split.get(0);

        int numBox = Math.toIntExact(box.getShape().get(0));

        float[] buf = box.toFloatArray();
        float[] confidences = split.get(1).max(AXIS_1).toFloatArray();
        long[] ids = split.get(1).argMax(1).toLongArray();

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
        long[] idx = nms.stream().mapToLong(Integer::longValue).toArray();
        NDArray selected = box.getManager().create(idx);
        NDArray masks = split.get(2).get(selected);

        int maskW = Math.toIntExact(protos.getShape().get(2));
        int maskH = Math.toIntExact(protos.getShape().get(1));

        protos = protos.reshape(32, (long) maskH * maskW);
        masks =
                masks.matMul(protos)
                        .reshape(nms.size(), maskH, maskW)
                        .gt(0f)
                        .toType(DataType.FLOAT32, true);

        float[] maskArray = masks.toFloatArray();
        box = box.get(selected);
        buf = box.toFloatArray();

        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();
        for (int i = 0; i < idx.length; ++i) {
            float x = buf[i * 4] / width;
            float y = buf[i * 4 + 1] / height;
            float w = buf[i * 4 + 2] / width - x;
            float h = buf[i * 4 + 3] / width - y;
            int id = nms.get(i);
            retClasses.add(classes.get((int) ids[id]));
            retProbs.add((double) confidences[id]);

            float[][] maskFloat = new float[maskH][maskW];
            int pos = i * maskH * maskW;
            for (int j = 0; j < maskH; j++) {
                System.arraycopy(maskArray, pos + j * maskW, maskFloat[j], 0, maskW);
            }
            Mask bb = new Mask(x, y, w, h, maskFloat, true);
            retBB.add(bb);
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@code YoloSegmentationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code YoloSegmentationTranslator} with specified arguments.
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

    /** The builder for instance segmentation translator. */
    public static class Builder extends YoloV5Translator.Builder {

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public YoloSegmentationTranslator build() {
            validate();
            return new YoloSegmentationTranslator(this);
        }
    }
}
