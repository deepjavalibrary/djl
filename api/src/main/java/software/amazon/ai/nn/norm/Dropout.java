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
package software.amazon.ai.nn.norm;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.AbstractBlock;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.util.PairList;

public class Dropout extends AbstractBlock {

    private static final byte VERSION = 1;

    private float probability;
    private int[] sharedAxes;

    Dropout(Builder builder) {
        probability = builder.getProbability();
        sharedAxes = builder.getSharedAxes();
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Dropout requires exactly 1 NDArray");
        }
        ensureInitialized(inputs);
        NDArrayEx ex = inputs.head().getNDArrayInternal();
        return ex.dropout(inputs, probability, sharedAxes, params);
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0];
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("Dropout has no parameters");
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

    public static final class Builder {

        private float probability = 0.5f;
        private int[] sharedAxes = {};

        public float getProbability() {
            return probability;
        }

        public int[] getSharedAxes() {
            return sharedAxes;
        }

        public Builder setProbability(float probability) {
            this.probability = probability;
            return this;
        }

        public Builder setSharedAxes(int[] sharedAxes) {
            this.sharedAxes = sharedAxes;
            return this;
        }

        public Dropout build() {
            return new Dropout(this);
        }
    }
}
