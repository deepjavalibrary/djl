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
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SimplePoseTranslator;
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
 * The translator for Simple Pose models.
 *
 * <p>The model was trained on Gluon and loaded in DJL in MXNet Symbol Block. See <a
 * href="https://arxiv.org/pdf/1804.06208.pdf">Simple Pose</a>.
 */
public class SimplePoseModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.CV.POSE_ESTIMATION;

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param groupId the group id of the model
     * @param artifactId the artifact id of the model
     * @param version the version number of the model
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    public SimplePoseModelLoader(
            Repository repository,
            String groupId,
            String artifactId,
            String version,
            ModelZoo modelZoo) {
        super(repository, MRL.model(APPLICATION, groupId, artifactId), version, modelZoo);
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(Image.class, Joints.class), factory);
        factories.put(new Pair<>(Path.class, Joints.class), new FileTranslatorFactory<>(factory));
        factories.put(new Pair<>(URL.class, Joints.class), new UrlTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(InputStream.class, Joints.class),
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
    public ZooModel<Image, Joints> loadModel()
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
    public ZooModel<Image, Joints> loadModel(Progress progress)
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
    public ZooModel<Image, Joints> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, Joints> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Joints.class)
                        .optModelZoo(modelZoo)
                        .optGroupId(resource.getMrl().getGroupId())
                        .optArtifactId(resource.getMrl().getArtifactId())
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, Joints> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, Joints> newInstance(Model model, Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 192d)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 256d)).intValue();
            double threshold = ((Double) arguments.getOrDefault("threshold", 0.2d));

            return SimplePoseTranslator.builder()
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .addTransform(new Normalize(MEAN, STD))
                    .optThreshold((float) threshold)
                    .build();
        }
    }
}
