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
package ai.djl.nn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * {@code LambdaBlock} is a {@link Block} with no parameters or children.
 *
 * <p>{@code LambdaBlock} allows converting any function that takes an {@code NDList} as input and
 * returns an {@code NDList} into a parameter-less and child-less {@link Block}. This can be useful
 * in converting activation functions, identity blocks, and more. For example, identity block can be
 * created as {@code new LambdaBlock(x -> x)}.
 */
public class LambdaBlock extends ParameterBlock {

    private static final byte VERSION = 1;

    private Function<NDList, NDList> lambda;

    /**
     * Creates a LambdaBlock that can apply the specified function.
     *
     * @param lambda the function to apply
     */
    public LambdaBlock(Function<NDList, NDList> lambda) {
        this.lambda = lambda;
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return lambda.apply(inputs);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        try (NDManager subManager = manager.newSubManager()) {
            NDList input = new NDList(inputShapes.length);
            for (Shape shape : inputShapes) {
                input.add(subManager.create(shape));
            }
            NDList output = lambda.apply(input);
            Shape[] outputShapes = new Shape[output.size()];
            for (int i = 0; i < output.size(); ++i) {
                outputShapes[i] = output.get(i).getShape();
            }
            return outputShapes;
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("LambdaBlocks have no parameters");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "Lambda()";
    }
}
