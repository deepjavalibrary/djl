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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.tensorflow.zoo.TfModelZoo;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;

/**
 * Model loader for Single Shot Detection models.
 *
 * <p>The model was trained on TensorFlow and loaded in DJL in {@link
 * ai.djl.tensorflow.engine.TfSymbolBlock}. See <a
 * href="https://arxiv.org/pdf/1512.02325.pdf">SSD</a>.
 *
 * <p>The model was obtained from <a
 * href="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1">TensorFlow Hub</a>
 *
 * @see ai.djl.tensorflow.engine.TfSymbolBlock
 */
public class SingleShotDetectionModelLoader extends BaseModelLoader<Image, DetectedObjects> {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = TfModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new TfModelZoo());
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(Image.class, DetectedObjects.class), factory);
        factories.put(
                new Pair<>(Path.class, DetectedObjects.class),
                new FileTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(URL.class, DetectedObjects.class), new UrlTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(InputStream.class, DetectedObjects.class),
                new InputStreamTranslatorFactory<>(factory));
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    /** {@inheritDoc} */
    @Override
    public ZooModel<Image, DetectedObjects> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, DetectedObjects> {

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public Translator<Image, DetectedObjects> newInstance(Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 256)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 256)).intValue();
            double threshold = (Double) arguments.getOrDefault("threshold", 0.4d);

            return TfSsdTranslator.builder()
                    .optMaxBoxes(10)
                    .optThreshold((float) threshold)
                    .optSynsetArtifactName("classes.txt")
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .build();
        }
    }
}
