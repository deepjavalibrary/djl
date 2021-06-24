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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SuperResolution {

    private static final Logger logger = LoggerFactory.getLogger(SuperResolution.class);

    private SuperResolution() {}

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        String imagePath = "src/test/resources/";
        ImageFactory imageFactory = ImageFactory.getInstance();

        Image[] inputImages = {imageFactory.fromFile(Paths.get(imagePath + "fox.png"))};
        Image[] enhancedImages = enhance(inputImages);

        if (enhancedImages == null) {
            logger.info("This example only works for TensorFlow Engine");
        } else {
            logger.info("Using TensorFlow Engine. {} images generated.", enhancedImages.length);
            saveImages(inputImages, enhancedImages);
        }
    }

    private static void saveImages(Image[] input, Image[] generated) throws IOException {
        Path outputPath = Paths.get("build/output/super-res/");
        Files.createDirectories(outputPath);

        save(generated, "image", outputPath);
        save(group(input, generated), "stitch", outputPath);

        logger.info("Generated images have been saved in: {}", outputPath);
    }

    private static void save(Image[] images, String name, Path path) throws IOException {
        for (int i = 0; i < images.length; i++) {
            Path imagePath = path.resolve(name + i + ".png");
            images[i].save(Files.newOutputStream(imagePath), "png");
        }
    }

    private static Image[] group(Image[] input, Image[] generated) {
        NDManager manager = Engine.getInstance().newBaseManager();
        NDList stitches = new NDList(input.length);

        int width = input[0].getWidth();
        int height = input[0].getHeight();
        for (int i = 0; i < input.length; i++) {
            NDArray left = input[i].toNDArray(manager);
            NDArray right = generated[i].toNDArray(manager);

            left = NDImageUtils.resize(left, 4 * width, 4 * height, Image.Interpolation.BICUBIC);
            right = right.toType(DataType.FLOAT32, false);

            stitches.add(NDArrays.concat(new NDList(left, right), 1));
        }

        return stitches.stream()
                .map(array -> array.toType(DataType.UINT8, false))
                .map(array -> ImageFactory.getInstance().fromNDArray(array))
                .toArray(Image[]::new);
    }

    public static Image[] enhance(Image[] inputImages)
            throws IOException, ModelException, TranslateException {
        if (!"TensorFlow".equals(Engine.getInstance().getEngineName())) {
            return null;
        }

        String modelUrl =
                "https://storage.googleapis.com/tfhub-modules/captain-pool/esrgan-tf2/1.tar.gz";
        Criteria<Image[], Image[]> criteria =
                Criteria.builder()
                        .setTypes(Image[].class, Image[].class)
                        .optModelUrls(modelUrl)
                        .optOption("Tags", "serve")
                        .optOption("SignatureDefKey", "serving_default")
                        .optTranslator(new SRTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image[], Image[]> model = criteria.loadModel();
                Predictor<Image[], Image[]> enhancer = model.newPredictor()) {
            return enhancer.predict(inputImages);
        }
    }
}
