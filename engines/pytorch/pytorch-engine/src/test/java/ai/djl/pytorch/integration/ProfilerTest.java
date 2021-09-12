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

package ai.djl.pytorch.integration;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.Test;

/** The file is for testing PyTorch Profiler functionalities. */
public class ProfilerTest {

    @Test
    public void testProfiler()
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageClassificationTranslator translator =
                    ImageClassificationTranslator.builder().addTransform(new ToTensor()).build();

            Criteria<Image, Classifications> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, Classifications.class)
                            .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                            .optFilter("layers", "18")
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .build();
            String outputFile = "build/profile.json";
            try (ZooModel<Image, Classifications> model = criteria.loadModel();
                    Predictor<Image, Classifications> predictor = model.newPredictor()) {
                Image image =
                        ImageFactory.getInstance()
                                .fromNDArray(manager.zeros(new Shape(3, 224, 224), DataType.UINT8));
                JniUtils.startProfile(false, true, true);
                predictor.predict(image);
                JniUtils.stopProfile(outputFile);
            }
            Assert.assertTrue(Files.exists(Paths.get(outputFile)), "The profiler file not found!");
        }
    }
}
