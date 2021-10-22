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
import java.util.PriorityQueue;

/**
 * A translator for YoloV5 models. This was tested with ONNX exported Yolo models. For details check
 * here: https://github.com/ultralytics/yolov5
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

    protected double boxIntersection(Rectangle a, Rectangle b) {
        double w =
                overlap(
                        (a.getX() * 2 + a.getWidth()) / 2,
                        a.getWidth(),
                        (b.getX() * 2 + b.getWidth()) / 2,
                        b.getWidth());
        double h =
                overlap(
                        (a.getY() * 2 + a.getHeight()) / 2,
                        a.getHeight(),
                        (b.getY() * 2 + b.getHeight()) / 2,
                        b.getHeight());
        if (w < 0 || h < 0) {
            return 0;
        }
        return w * h;
    }

    protected double boxIou(Rectangle a, Rectangle b) {
        return boxIntersection(a, b) / boxUnion(a, b);
    }

    protected double boxUnion(Rectangle a, Rectangle b) {
        double i = boxIntersection(a, b);
        return (a.getWidth()) * (a.getHeight()) + (b.getWidth()) * (b.getHeight()) - i;
    }

    protected DetectedObjects nms(List<IntermediateResult> list) {
        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int k = 0; k < classes.size(); k++) {
            // 1.find max confidence per class
            PriorityQueue<IntermediateResult> pq =
                    new PriorityQueue<>(
                            50,
                            (lhs, rhs) -> {
                                // Intentionally reversed to put high confidence at the head of the
                                // queue.
                                return Double.compare(rhs.getConfidence(), lhs.getConfidence());
                            });

            for (IntermediateResult intermediateResult : list) {
                if (intermediateResult.getDetectedClass() == k) {
                    pq.add(intermediateResult);
                }
            }

            // 2.do non maximum suppression
            while (pq.size() > 0) {
                // insert detection with max confidence
                IntermediateResult[] a = new IntermediateResult[pq.size()];
                IntermediateResult[] detections = pq.toArray(a);
                Rectangle rec = detections[0].getLocation();
                retClasses.add(detections[0].id);
                retProbs.add(detections[0].confidence);
                retBB.add(new Rectangle(rec.getX(), rec.getY(), rec.getWidth(), rec.getHeight()));
                pq.clear();
                for (int j = 1; j < detections.length; j++) {
                    IntermediateResult detection = detections[j];
                    Rectangle location = detection.getLocation();
                    if (boxIou(rec, location) < nmsThreshold) {
                        pq.add(detection);
                    }
                }
            }
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    protected double overlap(double x1, double w1, double x2, double w2) {
        double l1 = x1 - w1 / 2;
        double l2 = x2 - w2 / 2;
        double left = Math.max(l1, l2);
        double r1 = x1 + w1 / 2;
        double r2 = x2 + w2 / 2;
        double right = Math.min(r1, r2);
        return right - left;
    }

    private DetectedObjects processFromBoxOutput(NDList list) {
        float[] flattened = list.get(0).toFloatArray();
        ArrayList<IntermediateResult> intermediateResults = new ArrayList<>();
        int sizeClasses = classes.size();
        int stride = 5 + sizeClasses;
        int size = flattened.length / stride;
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
                intermediateResults.add(
                        new IntermediateResult(classes.get(maxIndex), score, maxIndex, rect));
            }
        }
        return nms(intermediateResults);
    }

    private DetectedObjects processFromDetectOutput() {
        throw new UnsupportedOperationException(
                "detect layer output is not supported yet, check correct YoloV5 export format");
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        switch (yoloOutputLayerType) {
            case DETECT:
                return processFromDetectOutput();
            case AUTO:
                if (list.get(0).getShape().dimension() > 2) {
                    return processFromDetectOutput();
                } else {
                    return processFromBoxOutput(list);
                }
            case BOX:
            default:
                return processFromBoxOutput(list);
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

    private static final class IntermediateResult {

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be
         * better.
         */
        private double confidence;

        /** Display name for the recognition. */
        private int detectedClass;

        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance
         * of the object.
         */
        private String id;

        /** Optional location within the source image for the location of the recognized object. */
        private Rectangle location;

        IntermediateResult(String id, double confidence, int detectedClass, Rectangle location) {
            this.confidence = confidence;
            this.id = id;
            this.detectedClass = detectedClass;
            this.location = location;
        }

        public double getConfidence() {
            return confidence;
        }

        public int getDetectedClass() {
            return detectedClass;
        }

        public String getId() {
            return id;
        }

        public Rectangle getLocation() {
            return new Rectangle(
                    location.getX(), location.getY(), location.getWidth(), location.getHeight());
        }
    }
}
