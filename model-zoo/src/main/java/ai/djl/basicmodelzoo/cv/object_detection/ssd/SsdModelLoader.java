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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.modality.cv.zoo.ObjectDetectionModelLoader;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Model loader for SingleShotDetection(SSD). */
public class SsdModelLoader extends ObjectDetectionModelLoader {

    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "ssd";
    private static final String VERSION = "0.0.2";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public SsdModelLoader(Repository repository) {
        super(repository, GROUP_ID, ARTIFACT_ID, VERSION, new BasicModelZoo());
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
}
