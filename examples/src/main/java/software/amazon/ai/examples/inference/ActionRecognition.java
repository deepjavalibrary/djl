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
package software.amazon.ai.examples.inference;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.zoo.ModelZoo;
import software.amazon.ai.examples.inference.util.AbstractExample;
import software.amazon.ai.examples.inference.util.Arguments;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.zoo.ModelNotFoundException;
import software.amazon.ai.zoo.ZooModel;

public class ActionRecognition extends AbstractExample {

    public static void main(String[] args) {
        new ActionRecognition().runExample(args);
    }

    @Override
    protected List<Classification> predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException {

        List<Classification> result;
        Path imageFile = arguments.getImageFile();
        BufferedImage img = Images.loadImageFromFile(imageFile);
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "inceptionv3");
        criteria.put("dataset", "ucf101");
        ZooModel<BufferedImage, List<Classification>> inception =
                ModelZoo.ACTION_RECOGNITION.loadModel(criteria);

        try (Predictor<BufferedImage, List<Classification>> action = inception.newPredictor()) {
            action.setMetrics(metrics); // Let predictor collect metrics
            result = action.predict(img);
            collectMemoryInfo(metrics);
        }

        inception.close();
        return result;
    }
}
