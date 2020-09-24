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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
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
 * A {@link ai.djl.repository.zoo.ModelLoader} for Single Shot Detection (SSD) models.
 *
 * <p>These models were built as part of the <a
 * href="https://gluon-cv.mxnet.io/model_zoo/detection.html">Gluon CV</a> library and imported into
 * DJL.
 *
 * <p>SSD is a model to solve {@link Application.CV#OBJECT_DETECTION}. Prior models before SSD
 * typically used non-deep learning strategies to find object proposals and then would execute
 * classification at those proposed locations. SSD integrated the proposal process into the model
 * and then classifies at all of those locations in parallel. This both simplifies the model and
 * training process while better leveraging the power of model DL engines. [<a
 * href="https://arxiv.org/pdf/1512.02325.pdf">paper</a>]
 *
 * <p>Today, SSD is not typically used in favor of newer models which outperform it.
 */
public class ObjectDetectionModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param groupId the group id of the model
     * @param artifactId the artifact id of the model
     * @param version the version number of the model
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    public ObjectDetectionModelLoader(
            Repository repository,
            String groupId,
            String artifactId,
            String version,
            ModelZoo modelZoo) {
        super(repository, MRL.model(APPLICATION, groupId, artifactId), version, modelZoo);
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

    /**
     * Loads the model.
     *
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<Image, DetectedObjects> loadModel()
            throws MalformedModelException, ModelNotFoundException, IOException {
        return loadModel(null, null, null);
    }

    /**
     * Loads the model.
     *
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<Image, DetectedObjects> loadModel(Progress progress)
            throws MalformedModelException, ModelNotFoundException, IOException {
        return loadModel(null, null, progress);
    }

    /**
     * Loads the model with the given search filters.
     *
     * @param filters the search filters to match against the loaded model
     * @param device the device the loaded model should use
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<Image, DetectedObjects> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelZoo(modelZoo)
                        .optGroupId(resource.getMrl().getGroupId())
                        .optArtifactId(resource.getMrl().getArtifactId())
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, DetectedObjects> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, DetectedObjects> newInstance(
                Model model, Map<String, ?> arguments) {
            return SingleShotDetectionTranslator.builder(arguments).build();
        }
    }
}
