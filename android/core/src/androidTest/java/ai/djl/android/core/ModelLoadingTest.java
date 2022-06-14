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
package ai.djl.android.core;

import org.junit.Test;

import java.io.IOException;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

/* Integration test to check if model and inference runs fine */
public class ModelLoadingTest {
        @Test
        public void testModelLoading() throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
            ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                    .addTransform(new ToTensor())
                    .optFlag(Image.Flag.GRAYSCALE)
                    .optApplySoftmax(true)
                    .build();
            Criteria<Image, Classifications> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, Classifications.class)
                            .optModelUrls("https://djl-ai.s3.amazonaws.com/resources/demo/pytorch/doodle_mobilenet.zip")
                            .optTranslator(translator)
                            .build();
            Image image = ImageFactory.getInstance().fromUrl("https://github.com/deepjavalibrary/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg");
            ZooModel<Image, Classifications> model =  ModelZoo.loadModel(criteria);
            Predictor<Image, Classifications> predictor = model.newPredictor();
            Classifications result = predictor.predict(image);
            System.out.println(result);
        }
}
