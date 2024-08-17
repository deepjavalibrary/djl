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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A translator for YoloV5 models. This was tested with ONNX exported Yolo models. For details check
 * <a href="https://github.com/ultralytics/yolov5">here</a>
 */
public class YoloV5Translator extends ObjectDetectionTranslator {

    private YoloOutputType yoloOutputLayerType;
    private float nmsThreshold;

    /**
     * Constructs an ImageTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    protected YoloV5Translator(Builder builder) {
        super(builder);
        yoloOutputLayerType = builder.outputType;
        nmsThreshold = builder.nmsThreshold;
    }

    /**
     * Creates a builder to build a {@link YoloV5Translator}.
     *
     * @return a new builder
     */
    public static YoloV5Translator.Builder builder() {
        return new YoloV5Translator.Builder();
    }

    /**
     * Creates a builder to build a {@code YoloV5Translator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static YoloV5Translator.Builder builder(Map<String, ?> arguments) {
        YoloV5Translator.Builder builder = new YoloV5Translator.Builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);
        return builder;
    }

    protected DetectedObjects nms(
            int imageWidth,
            int imageHeight,
            List<Rectangle> boxes,
            List<Integer> classIds,
            List<Float> scores) {
        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int classId = 0; classId < classes.size(); classId++) {
            List<Rectangle> r = new ArrayList<>();
            List<Double> s = new ArrayList<>();
            List<Integer> map = new ArrayList<>();
            for (int j = 0; j < classIds.size(); ++j) {
                if (classIds.get(j) == classId) {
                    r.add(boxes.get(j));
                    s.add(scores.get(j).doubleValue());
                    map.add(j);
                }
            }
            if (r.isEmpty()) {
                continue;
            }
            List<Integer> nms = Rectangle.nms(r, s, nmsThreshold);
            for (int index : nms) {
                int pos = map.get(index);
                int id = classIds.get(pos);
                retClasses.add(classes.get(id));
                retProbs.add(scores.get(pos).doubleValue());
                Rectangle rect = boxes.get(pos);
                if (removePadding) {
                    int padW = (width - imageWidth) / 2;
                    int padH = (height - imageHeight) / 2;
                    rect =
                            new Rectangle(
                                    (rect.getX() - padW) / imageWidth,
                                    (rect.getY() - padH) / imageHeight,
                                    rect.getWidth() / imageWidth,
                                    rect.getHeight() / imageHeight);
                } else if (applyRatio) {
                    rect =
                            new Rectangle(
                                    rect.getX() / width,
                                    rect.getY() / height,
                                    rect.getWidth() / width,
                                    rect.getHeight() / height);
                }
                retBB.add(rect);
            }
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    protected DetectedObjects processFromBoxOutput(int imageWidth, int imageHeight, NDList list) {
        float[] flattened = list.get(0).toFloatArray();
        int sizeClasses = classes.size();
        int stride = 5 + sizeClasses;
        int size = flattened.length / stride;

        ArrayList<Rectangle> boxes = new ArrayList<>();
        ArrayList<Float> scores = new ArrayList<>();
        ArrayList<Integer> classIds = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            int indexBase = i * stride;
            float maxClass = 0;
            int maxIndex = 0;
            for (int c = 0; c < sizeClasses; c++) {
                if (flattened[indexBase + c + 5] > maxClass) {
                    maxClass = flattened[indexBase + c + 5];
                    maxIndex = c;
                }
            }
            float score = maxClass * flattened[indexBase + 4];
            if (score > threshold) {
                float xPos = flattened[indexBase];
                float yPos = flattened[indexBase + 1];
                float w = flattened[indexBase + 2];
                float h = flattened[indexBase + 3];
                Rectangle rect =
                        new Rectangle(Math.max(0, xPos - w / 2), Math.max(0, yPos - h / 2), w, h);
                boxes.add(rect);
                scores.add(score);
                classIds.add(maxIndex);
            }
        }
        return nms(imageWidth, imageHeight, boxes, classIds, scores);
    }

    private DetectedObjects processFromDetectOutput() {
        throw new UnsupportedOperationException(
                "detect layer output is not supported yet, check correct YoloV5 export format");
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        int imageWidth = (Integer) ctx.getAttachment("width");
        int imageHeight = (Integer) ctx.getAttachment("height");
        switch (yoloOutputLayerType) {
            case DETECT:
                return processFromDetectOutput();
            case AUTO:
                if (list.get(0).getShape().dimension() > 2) {
                    return processFromDetectOutput();
                } else {
                    return processFromBoxOutput(imageWidth, imageHeight, list);
                }
            case BOX:
            default:
                return processFromBoxOutput(imageWidth, imageHeight, list);
        }
    }

    /** A enum represents the Yolo output type. */
    public enum YoloOutputType {
        BOX,
        DETECT,
        AUTO
    }

    /** The builder for {@link YoloV5Translator}. */
    public static class Builder extends ObjectDetectionBuilder<YoloV5Translator.Builder> {

        YoloOutputType outputType = YoloOutputType.AUTO;
        float nmsThreshold = 0.4f;

        /**
         * Sets the {@code YoloOutputType}.
         *
         * @param outputType the {@code YoloOutputType}
         * @return this builder
         */
        public Builder optOutputType(YoloOutputType outputType) {
            this.outputType = outputType;
            return this;
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
            super.configPostProcess(arguments);
            String type = ArgumentsUtil.stringValue(arguments, "outputType", "AUTO");
            outputType = YoloOutputType.valueOf(type.toUpperCase(Locale.ENGLISH));
            nmsThreshold = ArgumentsUtil.floatValue(arguments, "nmsThreshold", 0.4f);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public YoloV5Translator build() {
            // custom pipeline to match default YoloV5 input layer
            if (pipeline == null) {
                addTransform(
                        array -> array.transpose(2, 0, 1).toType(DataType.FLOAT32, false).div(255));
            }
            validate();
            return new YoloV5Translator(this);
        }
    }
}
