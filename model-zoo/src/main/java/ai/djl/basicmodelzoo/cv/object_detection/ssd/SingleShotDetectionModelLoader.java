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

package ai.djl.basicmodelzoo.cv.object_detection.ssd;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Model loader for SingleShotDetection(SSD). */
public class SingleShotDetectionModelLoader extends BaseModelLoader<Image, DetectedObjects> {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(
                repository,
                MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID),
                VERSION,
                new BasicModelZoo());
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

    @SuppressWarnings("unchecked")
    private Block customSSDBlock(Map<String, Object> arguments) {
        int numClasses = ((Double) arguments.get("outSize")).intValue();
        int numFeatures = ((Double) arguments.get("numFeatures")).intValue();
        boolean globalPool = (boolean) arguments.get("globalPool");
        int[] numFilters =
                ((List<Double>) arguments.get("numFilters"))
                        .stream()
                        .mapToInt(Double::intValue)
                        .toArray();
        List<Float> ratio =
                ((List<Double>) arguments.get("ratios"))
                        .stream()
                        .map(Double::floatValue)
                        .collect(Collectors.toList());
        List<List<Float>> sizes =
                ((List<List<Double>>) arguments.get("sizes"))
                        .stream()
                        .map(
                                size ->
                                        size.stream()
                                                .map(Double::floatValue)
                                                .collect(Collectors.toList()))
                        .collect(Collectors.toList());
        List<List<Float>> ratios = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ratios.add(ratio);
        }
        SequentialBlock baseBlock = new SequentialBlock();
        for (int numFilter : numFilters) {
            baseBlock.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }

        return SingleShotDetection.builder()
                .setNumClasses(numClasses)
                .setNumFeatures(numFeatures)
                .optGlobalPool(globalPool)
                .setRatios(ratios)
                .setSizes(sizes)
                .setBaseNetwork(baseBlock)
                .build();
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
        model.setBlock(customSSDBlock(arguments));
        return model;
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, DetectedObjects> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, DetectedObjects> newInstance(Map<String, Object> arguments) {
            return SingleShotDetectionTranslator.builder()
                    .addTransform(new ToTensor())
                    .optThreshold(((Double) arguments.get("threshold")).floatValue())
                    .build();
        }
    }
}
