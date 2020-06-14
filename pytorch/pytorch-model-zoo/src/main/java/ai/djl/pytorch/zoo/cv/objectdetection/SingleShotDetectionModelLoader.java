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
package ai.djl.pytorch.zoo.cv.objectdetection;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.pytorch.zoo.PtModelZoo;
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

/**
 * Model loader for Single Shot Detection models.
 *
 * <p>The model was trained on PyTorch and loaded in DJL in {@link
 * ai.djl.pytorch.engine.PtSymbolBlock}. See <a href="https://arxiv.org/pdf/1512.02325.pdf">SSD</a>.
 *
 * @see ai.djl.pytorch.engine.PtSymbolBlock
 */
public class SingleShotDetectionModelLoader extends BaseModelLoader<Image, DetectedObjects> {

    private static final Application APPLICATION = Application.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = PtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new PtModelZoo());
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
            int width = ((Double) arguments.getOrDefault("width", 300)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 300)).intValue();
            double threshold = (Double) arguments.getOrDefault("threshold", 0.4d);
            int figSize = ((Double) arguments.getOrDefault("size", 300)).intValue();
            int[] featSize;
            List<Double> list = (List<Double>) arguments.get("feat_size");
            if (list == null) {
                featSize = new int[] {38, 19, 10, 5, 3, 1};
            } else {
                featSize = list.stream().mapToInt(Double::intValue).toArray();
            }
            int[] steps;
            list = (List<Double>) arguments.get("steps");
            if (list == null) {
                steps = new int[] {8, 16, 32, 64, 100, 300};
            } else {
                steps = list.stream().mapToInt(Double::intValue).toArray();
            }

            int[] scale;
            list = (List<Double>) arguments.get("scale");
            if (list == null) {
                scale = new int[] {21, 45, 99, 153, 207, 261, 315};
            } else {
                scale = list.stream().mapToInt(Double::intValue).toArray();
            }

            int[][] aspectRatio;
            List<List<Double>> ratio = (List<List<Double>>) arguments.get("aspect_ratios");
            if (ratio == null) {
                aspectRatio = new int[][] {{2}, {2, 3}, {2, 3}, {2, 3}, {2}, {2}};
            } else {
                aspectRatio = new int[ratio.size()][];
                for (int i = 0; i < aspectRatio.length; ++i) {
                    aspectRatio[i] = ratio.get(i).stream().mapToInt(Double::intValue).toArray();
                }
            }

            return PtSSDTranslator.builder()
                    .setBoxes(figSize, featSize, steps, scale, aspectRatio)
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .addTransform(new Normalize(MEAN, STD))
                    .optSynsetArtifactName("classes.txt")
                    .optThreshold((float) threshold)
                    .build();
        }
    }
}
