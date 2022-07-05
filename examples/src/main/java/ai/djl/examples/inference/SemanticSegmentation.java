/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {

    private static final Logger logger = LoggerFactory.getLogger(SemanticSegmentationTranslator.class);

    private SemanticSegmentation() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        SemanticSegmentation.predict();
    }

    public static void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        URL url = new URL("https://djl-misc.s3.amazonaws.com/tmp/semantic_segmentation/ai/djl/pytorch/deeplab/deeplabv3_scripted.pt");
        try (InputStream in = url.openStream()) {
            Files.copy(in, Paths.get("src/test/resources/deeplabv3_scripted.pt"));
        }

        // get image dimensions
        final int height = img.getHeight();
        final int width = img.getWidth();

        // rgb coloring means and standard deviations for improved detection performance
        final float[] MEAN = {0.485f, 0.456f, 0.406f};
        final float[] STD = {0.229f, 0.224f, 0.225f};

        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.SEMANTIC_SEGMENTATION)
                        .setTypes(Image.class, Image.class)
                        .optModelPath(Paths.get("src/test/resources/deeplabv3_scripted.pt"))
                        .optTranslator(SemanticSegmentationTranslator.builder().addTransform(new Resize(width, height)).addTransform(new ToTensor()).addTransform(new Normalize(MEAN, STD)).optSynsetUrl("https://mlrepo.djl.ai/model/cv/instance_segmentation/ai/djl/mxnet/mask_rcnn/classes.txt").build())
                        .optEngine("PyTorch")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<Image, Image> model = criteria.loadModel()) {
            try (Predictor<Image, Image> predictor = model.newPredictor()) {
                Image semanticImage = predictor.predict(img);
                saveSemanticImage(semanticImage);
            }
        }
    }

    private static void saveSemanticImage(Image img) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve("semantic_instances.png");
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Segmentation result image has been saved in: {}", imagePath);
    }
}
