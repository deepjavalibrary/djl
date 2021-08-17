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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Collections;

/** Block to perform the Bert next-sentence-prediction task. */
public class BertNextSentenceBlock extends AbstractBlock {

    private Linear binaryClassifier;

    /** Creates a next sentence block. */
    public BertNextSentenceBlock() {
        binaryClassifier =
                addChildBlock(
                        "binaryClassifier", Linear.builder().setUnits(2).optBias(true).build());
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.inputNames = Collections.singletonList("pooledOutput");
        this.binaryClassifier.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return new NDList(binaryClassifier.forward(ps, inputs, training).head().logSoftmax(1));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {new Shape(inputShapes[0].get(0), 2)};
    }
}
