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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;

/** Block to perform the Bert next-sentence-prediction task. */
public class BertNextSentenceBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private final Linear binaryClassifier;

    /** Creates a next sentence block. */
    public BertNextSentenceBlock() {
        super(VERSION);
        this.binaryClassifier =
                addChildBlock(
                        "binaryClassifier",
                        Linear.builder().setOutChannels(2).optBias(true).build());
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.inputNames = Arrays.asList("pooledOutput");
        this.binaryClassifier.initialize(manager, dataType, inputShapes);
    }

    @Override
    public NDList forward(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        return forward(ps, inputs, training);
    }

    @Override
    public NDList forward(ParameterStore ps, NDList inputs, boolean training) {
        return new NDList(forward(ps, inputs.singletonOrThrow(), training));
    }

    /**
     * Applies the internal binary classifier.
     *
     * @param ps the parameter store
     * @param pooledOutput the pooled output of the bert models first token (CLS)
     * @param training true: apply dropout etc.
     * @return log probabilities for each batch whether the two sentences are consecutive or not.
     */
    public NDArray forward(ParameterStore ps, NDArray pooledOutput, boolean training) {
        return binaryClassifier
                .forward(ps, new NDList(pooledOutput), training)
                .head()
                .logSoftmax(1);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[] {new Shape(inputShapes[0].get(0), 2)};
    }
}
