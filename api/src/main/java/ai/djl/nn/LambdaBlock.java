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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.function.Function;

/**
 * {@code LambdaBlock} is a {@link Block} with no parameters or children.
 *
 * <p>{@code LambdaBlock} allows converting any function that takes an {@code NDList} as input and
 * returns an {@code NDList} into a parameter-less and child-less {@link Block}. This can be useful
 * in converting activation functions, identity blocks, and more. For example, identity block can be
 * created as {@code new LambdaBlock(x -> x)}.
 */
public class LambdaBlock extends AbstractBlock {

    private static final byte VERSION = 2;

    private Function<NDList, NDList> lambda;

    /**
     * Creates a LambdaBlock that can apply the specified function.
     *
     * @param lambda the function to apply
     */
    public LambdaBlock(Function<NDList, NDList> lambda) {
        super(VERSION);
        this.lambda = lambda;
    }

    /**
     * Creates a {@link LambdaBlock} for a singleton function.
     *
     * @param lambda a function accepting a singleton {@link NDList} and returning another sinleton
     *     {@link NDList}
     * @return a new {@link LambdaBlock} for the function
     */
    public static LambdaBlock singleton(Function<NDArray, NDArray> lambda) {
        return new LambdaBlock(arrays -> new NDList(lambda.apply(arrays.singletonOrThrow())));
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return lambda.apply(inputs);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDList input = new NDList(inputShapes.length);
            for (Shape shape : inputShapes) {
                input.add(manager.zeros(shape));
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
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version == VERSION) {
            readInputShapes(is);
        } else if (version != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "Lambda()";
    }
}
