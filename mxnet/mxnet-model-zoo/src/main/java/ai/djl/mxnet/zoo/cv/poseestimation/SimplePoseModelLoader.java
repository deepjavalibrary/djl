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
package ai.djl.mxnet.zoo.cv.poseestimation;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.FileTranslatorFactory;
import ai.djl.modality.cv.InputStreamTranslatorFactory;
import ai.djl.modality.cv.Joints;
import ai.djl.modality.cv.SimplePoseTranslator;
import ai.djl.modality.cv.UrlTranslatorFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Progress;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The translator for Simple Pose models.
 *
 * <p>The model was trained on Gluon and loaded in DJL in MXNet Symbol Block. See <a
 * href="https://arxiv.org/pdf/1804.06208.pdf">Simple Pose</a>.
 *
 * @see ai.djl.mxnet.engine.MxSymbolBlock
 */
public class SimplePoseModelLoader extends BaseModelLoader<BufferedImage, Joints> {

    private static final Application APPLICATION = Application.CV.POSE_ESTIMATION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "simple_pose";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SimplePoseModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        FactoryImpl factory = new FactoryImpl();

        Map<Type, TranslatorFactory<?, ?>> map = new ConcurrentHashMap<>();
        map.put(BufferedImage.class, factory);
        map.put(Path.class, new FileTranslatorFactory<>(factory));
        map.put(URL.class, new UrlTranslatorFactory<>(factory));
        map.put(InputStream.class, new InputStreamTranslatorFactory<>(factory));

        factories.put(Joints.class, map);
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
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
    @Override
    public ZooModel<BufferedImage, Joints> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<BufferedImage, Joints> criteria =
                Criteria.builder()
                        .setTypes(BufferedImage.class, Joints.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<BufferedImage, Joints> {

        @Override
        public Translator<BufferedImage, Joints> newInstance(Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 192d)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 256d)).intValue();
            double threshold = ((Double) arguments.getOrDefault("threshold", 0.2d));

            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(width, height))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {0.485f, 0.456f, 0.406f},
                                    new float[] {0.229f, 0.224f, 0.225f}));

            return SimplePoseTranslator.builder()
                    .setPipeline(pipeline)
                    .optThreshold((float) threshold)
                    .build();
        }
    }
}
