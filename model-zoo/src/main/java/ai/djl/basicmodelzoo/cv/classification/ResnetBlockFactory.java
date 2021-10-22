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
package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.translate.ArgumentsUtil;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** A {@link BlockFactory} class that creates {@link ResNetV1} block. */
public class ResnetBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
        int numLayers = ArgumentsUtil.intValue(arguments, "numLayers");
        long outSize = ArgumentsUtil.longValue(arguments, "outSize");

        @SuppressWarnings("unchecked")
        Shape shape =
                new Shape(
                        ((List<Double>) arguments.get("imageShape"))
                                .stream()
                                .mapToLong(Double::longValue)
                                .toArray());
        ResNetV1.Builder blockBuilder =
                ResNetV1.builder().setNumLayers(numLayers).setOutSize(outSize).setImageShape(shape);
        if (arguments.containsKey("batchNormMomentum")) {
            float batchNormMomentum = ArgumentsUtil.floatValue(arguments, "batchNormMomentum");
            blockBuilder.optBatchNormMomentum(batchNormMomentum);
        }
        return blockBuilder.build();
    }
}
