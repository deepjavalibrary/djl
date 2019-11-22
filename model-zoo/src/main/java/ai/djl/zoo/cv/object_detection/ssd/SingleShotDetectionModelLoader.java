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

package ai.djl.zoo.cv.object_detection.ssd;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.SingleShotDetectionTranslator;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.zoo.ModelZoo;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Model loader for SingleShotDetection(SSD). */
public class SingleShotDetectionModelLoader
        extends BaseModelLoader<BufferedImage, DetectedObjects> {
    private static final Anchor BASE_ANCHOR = MRL.Model.CV.OBJECT_DETECTION;
    private static final String GROUP_ID = ModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SingleShotDetectionModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public Translator<BufferedImage, DetectedObjects> getTranslator(Artifact artifact) {
        Map<String, Object> arguments = artifact.getArguments();
        Pipeline pipeline = new Pipeline();
        pipeline.add(new ToTensor());
        return new SingleShotDetectionTranslator.Builder()
                .setPipeline(pipeline)
                .setSynsetArtifactName("synset.txt")
                .optThreshold(((Double) arguments.get("threshold")).floatValue())
                .build();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    protected Model loadModel(Artifact artifact, Path modelPath, Device device)
            throws IOException, MalformedModelException {
        Map<String, Object> arguments = artifact.getArguments();
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
                new SingleShotDetection.Builder()
                        .setNumClasses(numClasses)
                        .setNumFeatures(numFeatures)
                        .optGlobalPool(globalPool)
                        .setRatios(ratios)
                        .setSizes(sizes)
                        .setBaseNetwork(baseBlock)
                        .build();

        Model model = Model.newInstance(device);
        model.setBlock(ssdBlock);
        model.load(modelPath, artifact.getName());
        return model;
    }
}
