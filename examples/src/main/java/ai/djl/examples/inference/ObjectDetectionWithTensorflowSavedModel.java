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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference of object detection using saved model from TensorFlow 2 Detection Model
 * Zoo.
 *
 * <p>Tested with EfficientDet, SSD MobileNet V2, Faster RCNN Inception Resnet V2 downloaded from <a
 * href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">here</a>
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/object_detection_with_tensorflow_saved_model.md">doc</a>
 * for information about this example.
 */
public final class ObjectDetectionWithTensorflowSavedModel {

    private static final Logger logger =
            LoggerFactory.getLogger(ObjectDetectionWithTensorflowSavedModel.class);

    private ObjectDetectionWithTensorflowSavedModel() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = ObjectDetectionWithTensorflowSavedModel.predict();
        if (detection == null) {
            logger.info("This example only works for TensorFlow Engine");
        } else {
            logger.info("{}", detection);
        }
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        if (!"TensorFlow".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String modelUrl =
                "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz";

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelUrls(modelUrl)
                        // saved_model.pb file is in the subfolder of the model archive file
                        .optModelName("ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
                        .optTranslator(new MyTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(img);
            saveBoundingBoxImage(img, detection);
            return detection;
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected-tensorflow-model-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        logger.info("Detected objects image has been saved in: {}", imagePath);
    }

    static Map<Integer, String> loadSynset() throws IOException {
        URL synsetUrl =
                new URL(
                        "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt");
        Map<Integer, String> map = new ConcurrentHashMap<>();
        int maxId = 0;
        try (InputStream is = synsetUrl.openStream();
                Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
            scanner.useDelimiter("item ");
            while (scanner.hasNext()) {
                String content = scanner.next();
                content = content.replaceAll("(\"|\\d)\\n\\s", "$1,");
                Item item = JsonUtils.GSON.fromJson(content, Item.class);
                map.put(item.id, item.displayName);
                if (item.id > maxId) {
                    maxId = item.id;
                }
            }
        }
        return map;
    }

    private static final class Item {
        int id;

        @SerializedName("display_name")
        String displayName;
    }

    private static final class MyTranslator implements Translator<Image, DetectedObjects> {

        private Map<Integer, String> classes;
        private int maxBoxes;
        private float threshold;

        MyTranslator() {
            maxBoxes = 10;
            threshold = 0.7f;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            // input to tf object-detection models is a list of tensors, hence NDList
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            // optionally resize the image for faster processing
            array = NDImageUtils.resize(array, 224);
            // tf object-detection models expect 8 bit unsigned integer tensor
            array = array.toType(DataType.UINT8, true);
            array = array.expandDims(0); // tf object-detection models expect a 4 dimensional input
            return new NDList(array);
        }

        @Override
        public void prepare(NDManager manager, Model model) throws IOException {
            if (classes == null) {
                classes = loadSynset();
            }
        }

        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            // output of tf object-detection models is a list of tensors, hence NDList in djl
            // output NDArray order in the list are not guaranteed

            int[] classIds = null;
            float[] probabilities = null;
            NDArray boundingBoxes = null;
            for (NDArray array : list) {
                if ("detection_boxes".equals(array.getName())) {
                    boundingBoxes = array.get(0);
                } else if ("detection_scores".equals(array.getName())) {
                    probabilities = array.get(0).toFloatArray();
                } else if ("detection_classes".equals(array.getName())) {
                    // class id is between 1 - number of classes
                    classIds = array.get(0).toType(DataType.INT32, true).toIntArray();
                }
            }
            Objects.requireNonNull(classIds);
            Objects.requireNonNull(probabilities);
            Objects.requireNonNull(boundingBoxes);

            List<String> retNames = new ArrayList<>();
            List<Double> retProbs = new ArrayList<>();
            List<BoundingBox> retBB = new ArrayList<>();

            // result are already sorted
            for (int i = 0; i < Math.min(classIds.length, maxBoxes); ++i) {
                int classId = classIds[i];
                double probability = probabilities[i];
                // classId starts from 1, -1 means background
                if (classId > 0 && probability > threshold) {
                    String className = classes.getOrDefault(classId, "#" + classId);
                    float[] box = boundingBoxes.get(i).toFloatArray();
                    float yMin = box[0];
                    float xMin = box[1];
                    float yMax = box[2];
                    float xMax = box[3];
                    Rectangle rect = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
                    retNames.add(className);
                    retProbs.add(probability);
                    retBB.add(rect);
                }
            }

            return new DetectedObjects(retNames, retProbs, retBB);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
