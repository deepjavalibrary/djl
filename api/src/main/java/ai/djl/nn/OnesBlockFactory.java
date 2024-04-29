/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn;

import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.util.PairList;
import ai.djl.util.Utils;

import java.nio.file.Path;
import java.util.Map;

/** A {@link BlockFactory} class that creates LambdaBlock. */
public class OnesBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments) {
        String shapes = ArgumentsUtil.stringValue(arguments, "block_shapes");
        String blockNames = ArgumentsUtil.stringValue(arguments, "block_names");
        PairList<DataType, Shape> pairs = Shape.parseShapes(shapes);
        String[] names;
        if (blockNames != null) {
            names = blockNames.split(",");
        } else {
            names = Utils.EMPTY_ARRAY;
        }

        return Blocks.onesBlock(pairs, names);
    }
}
