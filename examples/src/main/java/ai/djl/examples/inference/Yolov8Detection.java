/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV8TranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * An example of inference using an yolov8 model.
 */
public final class Yolov8Detection {

    private static final Logger logger = LoggerFactory.getLogger(Yolov8Detection.class);

    private Yolov8Detection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = Yolov8Detection.predict();
        logger.info("{}", detection);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        String classPath = System.getProperty("java.class.path");
        String pathSeparator = System.getProperty("path.separator");
        classPath = classPath.split(pathSeparator)[0];
        Path modelPath = Paths.get(classPath + "/yolov8n.onnx");
        Path imgPath = Paths.get(classPath + "/yolov8_test.jpg");
        Image img = ImageFactory.getInstance().fromFile(imgPath);

        Map<String, Object> arguments = new HashMap<>();
        arguments.put("width", Integer.valueOf(640));
        arguments.put("height", Integer.valueOf(640));
        arguments.put("resize", "true");
        arguments.put("toTensor", true);
        arguments.put("applyRatio", true);
        arguments.put("threshold", 0.6f);
        arguments.put("synsetFileName", "yolov8_synset.txt");

        YoloV8TranslatorFactory yoloV8TranslatorFactory = new YoloV8TranslatorFactory();
        Translator<Image, DetectedObjects> translator = yoloV8TranslatorFactory.newInstance(Image.class, DetectedObjects.class, null, arguments);

        Criteria<Image, DetectedObjects> criteria = Criteria.builder().setTypes(Image.class, DetectedObjects.class).optModelPath(modelPath).optEngine("OnnxRuntime").optTranslator(translator)
            .optProgress(new ProgressBar()).build();

        DetectedObjects detectedObjects = null;
        DetectedObject detectedObject = null;
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
          try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            Path outputPath = Paths.get(classPath + "/output");
            Files.createDirectories(outputPath);

            detectedObjects = predictor.predict(img);

            if (detectedObjects.getNumberOfObjects() > 0) {
              List<DetectedObject> detectedObjectList = detectedObjects.items();
              for (int i = 0; i < detectedObjectList.size(); i++) {
                detectedObject = detectedObjectList.get(i);
                BoundingBox boundingBox = detectedObject.getBoundingBox();
                Rectangle tectangle = boundingBox.getBounds();
                System.out.println(
                    detectedObject.getClassName() + " " + detectedObject.getProbability() + " " + tectangle.getX() + " " + tectangle.getY() + " " + tectangle.getWidth() + " " + tectangle.getHeight());
              }
              
              saveBoundingBoxImage(img.resize(640, 640, false), detectedObjects, outputPath, imgPath.toFile().getName());
            }
            
            return detectedObjects;
          }
        }
      
        return null;
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detectedObjects, Path outputPath, String outputFileName) throws IOException {
      img.drawBoundingBoxes(detectedObjects);

      Path imagePath = outputPath.resolve(outputFileName);
      img.save(Files.newOutputStream(imagePath), "png");
    }
}
