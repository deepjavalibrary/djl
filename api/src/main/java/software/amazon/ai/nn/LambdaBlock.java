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
package software.amazon.ai.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class LambdaBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private Function<NDList, NDList> lambda;

    public LambdaBlock(Function<NDList, NDList> lambda) {
        this.lambda = lambda;
        initialized = true;
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        return lambda.apply(inputs);
    }

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

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("LambdaBlocks have no parameters");
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
    }
}
