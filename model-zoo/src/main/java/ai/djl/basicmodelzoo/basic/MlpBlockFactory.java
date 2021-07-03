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
package ai.djl.basicmodelzoo.basic;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** A {@link BlockFactory} class that creates MLP block. */
public class MlpBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
        Double width = (Double) arguments.get("width");
        if (width == null) {
            width = 28d;
        }
        Double height = (Double) arguments.get("height");
        if (height == null) {
            height = 28d;
        }
        int input = width.intValue() * height.intValue();
        int output = ((Double) arguments.get("output")).intValue();
        int[] hidden =
                ((List<Double>) arguments.get("hidden"))
                        .stream()
                        .mapToInt(Double::intValue)
                        .toArray();

        return new Mlp(input, output, hidden);
    }
}
