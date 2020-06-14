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
package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1.Builder;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Artifact;
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
import java.util.List;
import java.util.Map;

/** Model loader for ResNet_V1. */
public class ResNetModelLoader extends BaseModelLoader<Image, Classifications> {

    private static final Application APPLICATION = Application.CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "resnet";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public ResNetModelLoader(Repository repository) {
        super(
                repository,
                MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID),
                VERSION,
                new BasicModelZoo());
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(Image.class, Classifications.class), factory);
        factories.put(
                new Pair<>(Path.class, Classifications.class),
                new FileTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(URL.class, Classifications.class), new UrlTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(InputStream.class, Classifications.class),
                new InputStreamTranslatorFactory<>(factory));
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
    public ZooModel<Image, Classifications> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private Block resnetBlock(Map<String, Object> arguments) {
        @SuppressWarnings("unchecked")
        Shape shape =
                new Shape(
                        ((List<Double>) arguments.get("imageShape"))
                                .stream()
                                .mapToLong(Double::longValue)
                                .toArray());
        Builder blockBuilder =
                ResNetV1.builder()
                        .setNumLayers((int) ((double) arguments.get("numLayers")))
                        .setOutSize((long) ((double) arguments.get("outSize")))
                        .setImageShape(shape);
        if (arguments.containsKey("batchNormMomentum")) {
            float batchNormMomentum = (float) ((double) arguments.get("batchNormMomentum"));
            blockBuilder.optBatchNormMomentum(batchNormMomentum);
        }
        return blockBuilder.build();
    }

    /** {@inheritDoc} */
    @Override
    protected Model createModel(
            String name,
            Device device,
            Artifact artifact,
            Map<String, Object> arguments,
            String engine) {
        Model model = Model.newInstance(name, device, engine);
        model.setBlock(resnetBlock(arguments));
        return model;
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, Classifications> {

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public Translator<Image, Classifications> newInstance(Map<String, Object> arguments) {
            List<Double> shape = (List<Double>) arguments.get("imageShape");
            int width = shape.get(2).intValue();
            int height = shape.get(1).intValue();

            return ImageClassificationTranslator.builder()
                    .addTransform(new CenterCrop())
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .build();
        }
    }
}
