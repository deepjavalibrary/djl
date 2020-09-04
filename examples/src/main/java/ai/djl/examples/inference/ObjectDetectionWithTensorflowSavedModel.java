package ai.djl.examples.inference;/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license"file accompanying this file. This file is distributed on an "AS IS"BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.ObjectDetectionTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * An example of inference of object detection using saved model from TensorFlow 2 Detection Model Zoo.
 *
 * <p>Tested with EfficientDet, SSD MobileNet V2, Faster RCNN Inception Resnet V2 downloaded from <a
 * href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">here</a>
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/object_detection_with_tensorflow_saved_model.md">doc</a>
 * for information about this example.
 */
public final class ObjectDetectionWithTensorflowSavedModel {

    private static final Logger logger = LoggerFactory.getLogger(ObjectDetectionWithTensorflowSavedModel.class);
    private static final List<String> MS_COCO_CLASSES = Arrays.asList("", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella", "", "", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "", "dining table", "", "", "toilet", "", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven");

    static {
        if (System.getProperty("ai.djl.repository.zoo.location") == null)
            System.setProperty("ai.djl.repository.zoo.location", "./models/");
    }

    private ObjectDetectionWithTensorflowSavedModel() {
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = ObjectDetectionWithTensorflowSavedModel.predict();
        logger.info("{}", detection);
    }

    public static Criteria<Image, DetectedObjects> getTFSavedModelObjectDetectionCriteria() throws IOException {
        return Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optModelName("saved_model")//folder containing saved_model.pb in local repository
                .optTranslator(
                        TFSavedModelObjectDetectionTranslator
                                .builder()
                                .optSynset(MS_COCO_CLASSES)
                                .build()
                )
                .optProgress(new ProgressBar())
                .build();
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);


        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(getTFSavedModelObjectDetectionCriteria())) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

                DetectedObjects detection = predictor.predict(img);
                saveBoundingBoxImage(img, detection);
                return detection;
            }
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        logger.info("Detected objects image has been saved in: {}", imagePath);
    }

    private static final class TFSavedModelObjectDetectionTranslator extends ObjectDetectionTranslator {

        /**
         * Creates the {@link TFSavedModelObjectDetectionTranslator} from the given builder.
         *
         * @param builder the builder for the translator
         */
        protected TFSavedModelObjectDetectionTranslator(BaseBuilder<?> builder) {
            super(builder);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            //input to tf object-detection models is a list of tensors, hence NDList
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            array = NDImageUtils.resize(array, 224)//optionally resize the image for faster processing
                    .toType(DataType.UINT8, true);//tf object-detection models expect 8 bit unsigned integer tensor
            array = array.expandDims(0);//tf object-detection models expect a 4 dimensional input
            return new NDList(array);
        }

        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            //output of tf object-detection models is a list of tensors, hence NDList in djl

            //detected class ids are stored at index 5 of the NDList
            List<String> classes = Arrays.stream(list.get(5).get(0).toArray())
                    .map(c -> super.classes.get(c.intValue()))
                    .collect(Collectors.toList());

            //detected confidence scores are stored at index 1 of the NDList
            List<Double> probs = Arrays.stream(list.get(1).get(0).toArray())
                    .map(Number::doubleValue)
                    .collect(Collectors.toList());

            //detected bounding boxes are stored at index 7 of the NDList
            List<BoundingBox> bboxes = IntStream.range(0, classes.size())
                    .mapToObj(i -> list.get(7).get(0).get(i))
                    .map(e -> new Rectangle(
                            e.getFloat(1),
                            e.getFloat(0),
                            e.getFloat(3) - e.getFloat(1),
                            e.getFloat(2) - e.getFloat(0)))
                    .collect(Collectors.toList());

            //filter out classes that has confidence score > 70%
            List<Integer> indices = IntStream.range(0, probs.size())
                    .filter(i -> probs.get(i) < 0.7)
                    .boxed()
                    .collect(Collectors.toList());
            Collections.reverse(indices);
            for (int index : indices) {
                classes.remove(index);
                probs.remove(index);
                bboxes.remove(index);
            }

            return new DetectedObjects(classes, probs, bboxes);
        }


        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        /**
         * Creates a builder to build a {@code ImageClassificationTranslator}.
         *
         * @return a new builder
         */
        public static Builder builder() {
            return new Builder();
        }

        /**
         * A Builder to construct a {@code ImageClassificationTranslator}.
         */
        public static class Builder extends BaseBuilder<Builder> {

            private boolean applySoftmax;

            Builder() {
            }

            /**
             * Sets whether to apply softmax when processing output. Some models already include softmax in the last
             * layer, so don't apply softmax when processing model output.
             *
             * @param applySoftmax boolean whether to apply softmax
             * @return the builder
             */
            public Builder optApplySoftmax(boolean applySoftmax) {
                this.applySoftmax = applySoftmax;
                return this;
            }

            /**
             * {@inheritDoc}
             */
            @Override
            protected Builder self() {
                return this;
            }

            /**
             * Builds the {@link ImageClassificationTranslator} with the provided data.
             *
             * @return an {@link ImageClassificationTranslator}
             */
            public TFSavedModelObjectDetectionTranslator build() {
                addTransform(ndArray -> ndArray);//adding a dummy pipeline, execution fails without this
                validate();
                return new TFSavedModelObjectDetectionTranslator(this);
            }
        }
    }
}
