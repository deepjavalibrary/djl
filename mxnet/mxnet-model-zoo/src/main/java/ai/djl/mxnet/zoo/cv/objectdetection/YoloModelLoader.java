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
package ai.djl.mxnet.zoo.cv.objectdetection;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloTranslator;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
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
 * A {@link ai.djl.repository.zoo.ModelLoader} for YOLO models.
 *
 * <p>These models were built as part of the <a
 * href="https://gluon-cv.mxnet.io/model_zoo/detection.html">Gluon CV</a> library and imported into
 * DJL.
 *
 * <p>Yolo is a model to solve {@link Application.CV#OBJECT_DETECTION}. Prior models like {@link
 * SingleShotDetectionModelLoader} were built around classifiers and would classify at various
 * locations in the image simultaneously. However, this is fairly inefficient. Yolo instead uses a
 * regression that will predict both the bounding boxes and class probabilities leading to better
 * performance and better precision, although it can increase localization errors (the boxes are
 * less accurate). [<a href="https://arxiv.org/abs/1506.02640">paper</a>]
 *
 * <p>YOLO is currently the best object detection model in the DJL Model Zoo in terms of both
 * performance and prediction quality.
 */
public class YoloModelLoader extends BaseModelLoader<Image, DetectedObjects> {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "yolo";
    private static final String VERSION = "0.0.1";

    /**
     * Constructs a {@link YoloModelLoader} given the repository, mrl, and version.
     *
     * @param repository the repository to load the model from
     */
    public YoloModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new MxModelZoo());
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

    @Override
    public Application getApplication() {
        return APPLICATION;
    }

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

        @Override
        public Translator<Image, DetectedObjects> newInstance(Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 512d)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 512d)).intValue();
            double threshold = ((Double) arguments.getOrDefault("threshold", 0.2d));

            return YoloTranslator.builder()
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .optSynsetArtifactName("classes.txt")
                    .optThreshold((float) threshold)
                    .optRescaleSize(width, height)
                    .build();
        }
    }
}
