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
package software.amazon.ai.translate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import software.amazon.ai.ndarray.NDList;

public class Pipeline extends TransformBlock {

    private List<TransformBlock> transformBlocks;

    public Pipeline() {
        transformBlocks = new ArrayList<>();
    }

    public Pipeline(Transform... transforms) {
        transformBlocks =
                Arrays.stream(transforms).map(TransformBlock::new).collect(Collectors.toList());
    }

    public Pipeline(TransformBlock... transformBlocks) {
        this.transformBlocks = Arrays.asList(transformBlocks);
    }

    public Pipeline add(Transform transform) {
        transformBlocks.add(new TransformBlock(transform));
        return this;
    }

    public Pipeline add(TransformBlock transformBlock) {
        transformBlocks.add(transformBlock);
        return this;
    }

    @Override
    public NDList transform(NDList input, boolean close) {
        if (transformBlocks.isEmpty()) {
            return input;
        }

        NDList ret = transformBlocks.get(0).transform(input, close);
        for (int i = 1; i < transformBlocks.size(); ++i) {
            ret = transformBlocks.get(i).transform(ret, true);
        }
        return ret;
    }
}
