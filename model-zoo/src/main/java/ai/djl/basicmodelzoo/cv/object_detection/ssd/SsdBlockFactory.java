/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.nn.SequentialBlock;
import ai.djl.translate.ArgumentsUtil;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** A {@link BlockFactory} class that creates {@link SingleShotDetection} block. */
public class SsdBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
        int numClasses = ArgumentsUtil.intValue(arguments, "outSize");
        int numFeatures = ArgumentsUtil.intValue(arguments, "numFeatures");
        boolean globalPool = ArgumentsUtil.booleanValue(arguments, "globalPool");
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
