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
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.SingleShotDetectionTranslator;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/** Model loader for SingleShotDetection(SSD). */
public class SingleShotDetectionModelLoader extends BaseModelLoader {

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
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        Map<Type, TranslatorFactory<?, ?>> map = new ConcurrentHashMap<>();
        map.put(DetectedObjects.class, new FactoryImpl());
        factories.put(BufferedImage.class, map);
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public Model loadModel(Artifact artifact, Device device, Map<String, Object> override)
            throws IOException, MalformedModelException {
        Map<String, Object> arguments = artifact.getArguments(override);

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

        Block ssdBlock =
                SingleShotDetection.builder()
                        .setNumClasses(numClasses)
                        .setNumFeatures(numFeatures)
                        .optGlobalPool(globalPool)
                        .setRatios(ratios)
                        .setSizes(sizes)
                        .setBaseNetwork(baseBlock)
                        .build();

        Path dir = repository.getCacheDirectory();
        String relativePath = artifact.getResourceUri().getPath();
        Path modelPath = dir.resolve(relativePath);

        Model model = Model.newInstance(device);
        model.setBlock(ssdBlock);
        model.load(modelPath, artifact.getName());
        return model;
    }

    private static final class FactoryImpl
            implements TranslatorFactory<BufferedImage, DetectedObjects> {

        @Override
        public Translator<BufferedImage, DetectedObjects> newInstance(
                Map<String, Object> arguments) {
            Pipeline pipeline = new Pipeline();
            pipeline.add(new ToTensor());
            return SingleShotDetectionTranslator.builder()
                    .setPipeline(pipeline)
                    .setSynsetArtifactName("synset.txt")
                    .optThreshold(((Double) arguments.get("threshold")).floatValue())
                    .build();
        }
    }
}
