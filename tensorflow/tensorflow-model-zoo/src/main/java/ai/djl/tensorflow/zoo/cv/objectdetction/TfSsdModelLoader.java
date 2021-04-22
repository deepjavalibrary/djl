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
package ai.djl.tensorflow.zoo.cv.objectdetction;

import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.zoo.ObjectDetectionModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import java.util.Map;

/**
 * Model loader for Single Shot Detection models.
 *
 * <p>The model was trained on TensorFlow and loaded in DJL. See <a
 * href="https://arxiv.org/pdf/1512.02325.pdf">SSD</a>.
 *
 * <p>The model was obtained from <a
 * href="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1">TensorFlow Hub</a>
 *
 * @see ai.djl.nn.SymbolBlock
 */
public class TfSsdModelLoader extends ObjectDetectionModelLoader {

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param groupId the group id of the model
     * @param artifactId the artifact id of the model
     * @param version the version number of the model
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    public TfSsdModelLoader(
            Repository repository,
            String groupId,
            String artifactId,
            String version,
            ModelZoo modelZoo) {
        super(repository, groupId, artifactId, version, modelZoo);
        // override TranslatorFactory
        factories.put(new Pair<>(Image.class, DetectedObjects.class), new FactoryImpl());
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, DetectedObjects> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, DetectedObjects> newInstance(
                Model model, Map<String, ?> arguments) {

            return TfSsdTranslator.builder(arguments).build();
        }
    }
}
