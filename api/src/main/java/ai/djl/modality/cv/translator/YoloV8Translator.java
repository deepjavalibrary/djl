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

import java.util.ArrayList;
import java.util.Map;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

/**
 * A translator for YoloV8 models. This was tested with ONNX exported Yolo models. For details check here: https://github.com/ultralytics/ultralytics
 */
public class YoloV8Translator extends YoloV5Translator {

  /**
   * Constructs an ImageTranslator with the provided builder.
   *
   * @param builder the data to build with
   */
  protected YoloV8Translator(Builder builder) {
    super(builder);
  }

  /**
   * Creates a builder to build a {@code YoloV8Translator} with specified arguments.
   *
   * @param arguments arguments to specify builder options
   * @return a new builder
   */
  public static YoloV8Translator.Builder builder(Map<String, ?> arguments) {
    YoloV8Translator.Builder builder = new YoloV8Translator.Builder();
    builder.configPreProcess(arguments);
    builder.configPostProcess(arguments);

    return builder;
  }

  @Override
  protected DetectedObjects processFromBoxOutput(NDList list) {
    NDArray features4OneImg = list.get(0);
    int sizeClasses = classes.size();
    long sizeBoxes = features4OneImg.size(1);
    ArrayList<IntermediateResult> intermediateResults = new ArrayList<>();

    for (long b = 0; b < sizeBoxes; b++) {
      float maxClass = 0;
      int maxIndex = 0;
      for (int c = 4; c < sizeClasses; c++) {
        float classProb = features4OneImg.getFloat(c, b);
        if (classProb > maxClass) {
          maxClass = classProb;
          maxIndex = c;
        }
      }

      if (maxClass > threshold) {
        float xPos = features4OneImg.getFloat(0, b);// center x
        float yPos = features4OneImg.getFloat(1, b);// center y
        float w = features4OneImg.getFloat(2, b);
        float h = features4OneImg.getFloat(3, b);
        Rectangle rect = new Rectangle(Math.max(0, xPos - w / 2), Math.max(0, yPos - h / 2), w, h);
        intermediateResults.add(new IntermediateResult(classes.get(maxIndex), maxClass, maxIndex, rect));
      }
    }

    return nms(intermediateResults);
  }

  /** The builder for {@link YoloV8Translator}. */
  public static class Builder extends YoloV5Translator.Builder {
    /**
     * Builds the translator.
     *
     * @return the new translator
     */
    public YoloV8Translator build() {
      if (pipeline == null) {
        addTransform(array -> array.transpose(2, 0, 1).toType(DataType.FLOAT32, false).div(255));
      }
      validate();
      return new YoloV8Translator(this);
    }
  }
}
