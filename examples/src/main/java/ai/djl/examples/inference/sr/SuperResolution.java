/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.sr;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SuperResolution {

    private static final Logger logger = LoggerFactory.getLogger(SuperResolution.class);

    private SuperResolution() {}

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        String imagePath = "src/test/resources/";
        ImageFactory imageFactory = ImageFactory.getInstance();

        List<Image> inputImages =
                Arrays.asList(imageFactory.fromFile(Paths.get(imagePath + "fox.png")));

        List<Image> enhancedImages = enhance(inputImages);

        if (enhancedImages == null) {
            logger.info("This example only works for TensorFlow Engine");
        } else {
            logger.info("Using TensorFlow Engine. {} images generated.", enhancedImages.size());
            saveImages(inputImages, enhancedImages);
        }
    }

    private static void saveImages(List<Image> input, List<Image> generated) throws IOException {
        Path outputPath = Paths.get("build/output/super-res/");
        Files.createDirectories(outputPath);

        save(generated, "image", outputPath);
        save(group(input, generated), "stitch", outputPath);

        logger.info("Generated images have been saved in: {}", outputPath);
    }

    private static void save(List<Image> images, String name, Path path) throws IOException {
        for (int i = 0; i < images.size(); i++) {
            Path imagePath = path.resolve(name + i + ".png");
            images.get(i).save(Files.newOutputStream(imagePath), "png");
        }
    }

    private static List<Image> group(List<Image> input, List<Image> generated) {
        NDList stitches = new NDList(input.size());

        try (NDManager manager = Engine.getInstance().newBaseManager()) {
            for (int i = 0; i < input.size(); i++) {
                int scale = 4;
                int width = scale * input.get(i).getWidth();
                int height = scale * input.get(i).getHeight();

                NDArray left = input.get(i).toNDArray(manager);
                NDArray right = generated.get(i).toNDArray(manager);

                left = NDImageUtils.resize(left, width, height, Image.Interpolation.BICUBIC);
                right = right.toType(DataType.FLOAT32, false);

                stitches.add(NDArrays.concat(new NDList(left, right), 1));
            }

            return stitches.stream()
                    .map(array -> array.toType(DataType.UINT8, false))
                    .map(array -> ImageFactory.getInstance().fromNDArray(array))
                    .collect(Collectors.toList());
        }
    }

    public static List<Image> enhance(List<Image> inputImages)
            throws IOException, ModelException, TranslateException {

        if (!"TensorFlow".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelUrl =
                "https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz";
        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_ENHANCEMENT)
                        .setTypes(Image.class, Image.class)
                        .optModelUrls(modelUrl)
                        .optOption("Tags", "serve")
                        .optOption("SignatureDefKey", "serving_default")
                        .optTranslator(new SuperResolutionTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, Image> model = criteria.loadModel();
                Predictor<Image, Image> enhancer = model.newPredictor()) {
            return enhancer.batchPredict(inputImages);
        }
    }
}
