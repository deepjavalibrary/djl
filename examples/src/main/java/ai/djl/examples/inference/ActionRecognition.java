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
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference using an action recognition model.
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/action_recognition.md">doc</a> for
 * information about this example.
 */
public final class ActionRecognition {

    private static final Logger logger = LoggerFactory.getLogger(ActionRecognition.class);

    private ActionRecognition() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Classifications classification = ActionRecognition.predict();
        logger.info("{}", classification);
    }

    public static Classifications predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/action_discus_throw.png");
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        Criteria<BufferedImage, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.ACTION_RECOGNITION)
                        .setTypes(BufferedImage.class, Classifications.class)
                        .optOption("backbone", "inceptionv3")
                        .optOption("dataset", "ucf101")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<BufferedImage, Classifications> inception = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, Classifications> action = inception.newPredictor()) {
                return action.predict(img);
            }
        }
    }
}
