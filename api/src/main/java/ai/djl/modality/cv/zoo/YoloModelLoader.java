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
package ai.djl.modality.cv.zoo;

import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloTranslator;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import java.util.Map;

/**
 * A {@link ai.djl.repository.zoo.ModelLoader} for YOLO models.
 *
 * <p>These models were built as part of the <a
 * href="https://gluon-cv.mxnet.io/model_zoo/detection.html">Gluon CV</a> library and imported into
 * DJL.
 *
 * <p>Yolo is a model to solve {@link ai.djl.Application.CV#OBJECT_DETECTION}. Prior models like
 * {@link ObjectDetectionModelLoader} were built around classifiers and would classify at various
 * locations in the image simultaneously. However, this is fairly inefficient. Yolo instead uses a
 * regression that will predict both the bounding boxes and class probabilities leading to better
 * performance and better precision, although it can increase localization errors (the boxes are
 * less accurate). [<a href="https://arxiv.org/abs/1506.02640">paper</a>]
 *
 * <p>YOLO is currently the best object detection model in the DJL Model Zoo in terms of both
 * performance and prediction quality.
 */
public class YoloModelLoader extends ObjectDetectionModelLoader {

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param groupId the group id of the model
     * @param artifactId the artifact id of the model
     * @param version the version number of the model
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    public YoloModelLoader(
            Repository repository,
            String groupId,
            String artifactId,
            String version,
            ModelZoo modelZoo) {
        super(repository, groupId, artifactId, version, modelZoo);
        factories.put(new Pair<>(Image.class, DetectedObjects.class), new FactoryImpl());
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, DetectedObjects> {

        @Override
        public Translator<Image, DetectedObjects> newInstance(
                Model model, Map<String, ?> arguments) {
            return YoloTranslator.builder(arguments).build();
        }
    }
}
