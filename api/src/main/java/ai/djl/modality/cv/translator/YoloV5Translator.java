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
package ai.djl.modality.cv.translator;

import java.util.*;
import java.awt.geom.Rectangle2D;
import java.awt.geom.Rectangle2D.Double;
import ai.djl.modality.cv.output.*;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.*;

/**
 A translator for yolo V5 models. This was tested with ONNX exported Yolo models.
 For details check here: https://github.com/ultralytics/yolov5
 */
public class YoloV5Translator extends ObjectDetectionTranslator {
   private final YOLO_OUTPUT_TYPE yoloOutputLayerType;
   private double nmsThresh = 0.4F;

   public enum YOLO_OUTPUT_TYPE {
      BOX, DETECT, AUTO
   }

   /**
    Constructs an ImageTranslator with the provided builder.

    @param builder the data to build with
    */
   public YoloV5Translator(Builder builder) {
      this(builder, YOLO_OUTPUT_TYPE.AUTO);
   }

   protected YoloV5Translator(Builder builder, YOLO_OUTPUT_TYPE yoloOutputLayerType) {
      super(builder);
      this.yoloOutputLayerType = yoloOutputLayerType;
   }

   /**
    Creates a builder to build a {@link YoloV5Translator}.

    @return a new builder
    */
   public static YoloV5Translator.Builder builder() {
      return new YoloV5Translator.Builder();
   }

   /**
    Creates a builder to build a {@code YoloV5Translator} with specified arguments.

    @param arguments arguments to specify builder options
    @return a new builder
    */
   public static YoloV5Translator.Builder builder(Map<String, ?> arguments) {
      YoloV5Translator.Builder builder = new YoloV5Translator.Builder();
      builder.configPreProcess(arguments);
      builder.configPostProcess(arguments);

      return builder;
   }

   protected double boxIntersection(Rectangle2D a, Rectangle2D b) {
      double w = overlap((a.getMinX() + a.getMaxX()) / 2, a.getMaxX() - a.getMinX(),
            (b.getMinX() + b.getMaxX()) / 2, b.getMaxX() - b.getMinX());
      double h = overlap((a.getMinY() + a.getMaxY()) / 2, a.getMaxY() - a.getMinY(),
            (b.getMinY() + b.getMaxY()) / 2, b.getMaxY() - b.getMinY());
      if (w < 0 || h < 0) return 0;
      return w * h;
   }

   protected double boxIou(Rectangle2D a, Rectangle2D b) {
      return boxIntersection(a, b) / boxUnion(a, b);
   }

   protected double boxUnion(Rectangle2D a, Rectangle2D b) {
      double i = boxIntersection(a, b);
      return (a.getMaxX() - a.getMinX()) * (a.getMaxY() - a.getMinY()) + (b.getMaxX() - b.getMinX()) * (b.getMaxY() - b.getMinY()) - i;
   }

   public double getNmsThresh() {
      return nmsThresh;
   }

   public void setNmsThresh(double nmsThresh) {
      this.nmsThresh = nmsThresh;
   }

   protected DetectedObjects nms(ArrayList<IntermediateResult> list) {
      List<String> retClasses = new ArrayList<>();
      List<java.lang.Double> retProbs = new ArrayList<>();
      List<BoundingBox> retBB = new ArrayList<>();

      for (int k = 0; k < classes.size(); k++) {
         //1.find max confidence per class
         PriorityQueue<IntermediateResult> pq =
               new PriorityQueue<>(
                     50,
                     (lhs, rhs) -> {
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return java.lang.Double.compare(rhs.getConfidence(), lhs.getConfidence());
                     });

         for (IntermediateResult intermediateResult : list) {
            if (intermediateResult.getDetectedClass() == k) {
               pq.add(intermediateResult);
            }
         }

         //2.do non maximum suppression
         while (pq.size() > 0) {
            //insert detection with max confidence
            IntermediateResult[] a = new IntermediateResult[pq.size()];
            IntermediateResult[] detections = pq.toArray(a);
            final Rectangle2D rec = detections[0].getLocation();
            retClasses.add(detections[0].id);
            retProbs.add(detections[0].confidence);
            retBB.add(new Rectangle(rec.getX(), rec.getY(), rec.getWidth(), rec.getHeight()));
            pq.clear();
            for (int j = 1; j < detections.length; j++) {
               IntermediateResult detection = detections[j];
               final Rectangle2D location = detection.getLocation();
               if (boxIou(rec, location) < nmsThresh) {
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

   private DetectedObjects processFromBoxOutput(TranslatorContext ctx, NDList list) {
      final NDArray ndArray = list.get(0);
      final ArrayList<IntermediateResult> intermediateResults = new ArrayList<>();
      for (long i = 0; i < ndArray.size(0); i++) {
         final float[] boxes = ndArray.get(i).toFloatArray();
         float maxClass = 0;
         int maxIndex = 0;
         final float[] clazzes = new float[classes.size()];
         System.arraycopy(boxes, 5, clazzes, 0, clazzes.length);
         for (int c = 0; c < clazzes.length; c++) {
            if (clazzes[c] > maxClass) {
               maxClass = clazzes[c];
               maxIndex = c;
            }
         }
         final float score = maxClass * boxes[4];
         if (score > threshold) {
            final float xPos = boxes[0];
            final float yPos = boxes[1];
            final float w = boxes[2];
            final float h = boxes[3];
            intermediateResults.add(new IntermediateResult(classes.get(maxIndex), score, maxIndex, new Double(Math.max(0, xPos - w / 2), Math.max(0, yPos - h / 2), w, h)));
         }
      }
      return nms(intermediateResults);
   }

   public DetectedObjects processFromDetectOutput(TranslatorContext ctx, NDList list) {
      throw new RuntimeException("detect layer output is not supported yet, check correct YoloV5 export format");
   }

   /**
    {@inheritDoc}
    */
   @Override
   public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
      switch (yoloOutputLayerType) {
         case BOX:
            return processFromBoxOutput(ctx, list);
         case DETECT:
            return processFromDetectOutput(ctx, list);
         case AUTO:
            if (list.get(0).getShape().dimension() > 2) {
               return processFromDetectOutput(ctx, list);
            }
      }
      return processFromBoxOutput(ctx, list);
   }

   /**
    The builder for {@link YoloV5Translator}.
    */
   public static class Builder extends ObjectDetectionBuilder<YoloV5Translator.Builder> {
      public Builder() {
         // custom pipeline to match default YoloV5 input layer
         pipeline = new Pipeline().add(array -> array.transpose(2, 0, 1).toType(DataType.FLOAT32, false).div(255));
      }

      /**
       Builds the translator.

       @return the new translator
       */
      public YoloV5Translator build() {
         validate();
         return new YoloV5Translator(this);
      }

      /**
       {@inheritDoc}
       */
      @Override
      protected YoloV5Translator.Builder self() {
         return this;
      }
   }

   private static class IntermediateResult {
      /**
       A sortable score for how good the recognition is relative to others. Higher should be better.
       */
      private final double confidence;
      /**
       Display name for the recognition.
       */
      private final int detectedClass;
      /**
       A unique identifier for what has been recognized. Specific to the class, not the instance of
       the object.
       */
      private final String id;
      /**
       Optional location within the source image for the location of the recognized object.
       */
      private final Double location;

      private IntermediateResult(String id, double confidence, int detectedClass, Double location) {
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

      public Rectangle2D getLocation() {
         return new Double(location.x, location.y, location.width, location.height);
      }
   }

}
