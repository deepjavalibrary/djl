/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * A simpler version of the {@link PaddingStackBatchifier} that pads all dimensions rather than
 * specific ones.
 */
public final class SimplePaddingStackBatchifier implements Batchifier {

    private float padding;

    /**
     * A simple {@link Batchifier} that pads all arrays to same shape and then stacks them.
     *
     * @param padding the number of pad with
     */
    public SimplePaddingStackBatchifier(float padding) {
        this.padding = padding;
    }

    /**
     * A simple {@link Batchifier} that pads all arrays to same shape (with padding 0) and then
     * stacks them.
     */
    public SimplePaddingStackBatchifier() {
        this(0f);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchify(NDList[] inputs) {
        int numArrays = inputs[0].size();
        for (int i = 0; i < numArrays; i++) {
            int axes = inputs[0].get(i).getShape().dimension();
            for (int j = 0; j < axes; j++) {
                long maxSize = PaddingStackBatchifier.findMaxSize(inputs, i, j);
                NDManager manager = inputs[0].getManager();
                NDArray padArray = manager.create(padding);
                PaddingStackBatchifier.padArrays(inputs, i, j, padArray, maxSize);
            }
        }
        return Batchifier.STACK.batchify(inputs);
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] unbatchify(NDList inputs) {
        return Batchifier.STACK.unbatchify(inputs);
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        return Batchifier.STACK.split(list, numOfSlices, evenSplit);
    }
}
