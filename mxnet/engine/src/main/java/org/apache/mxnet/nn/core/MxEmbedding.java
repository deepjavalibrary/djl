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
package org.apache.mxnet.nn.core;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.core.Embedding;
import software.amazon.ai.util.PairList;

public class MxEmbedding<T> extends MxNNBlock implements Embedding<T> {

    private static final byte VERSION = 1;

    private int embeddingSize;
    private boolean useDefault;
    private DataType dataType;
    private Map<T, Integer> embedder;
    private int numItems;

    private Parameter embedding;

    public MxEmbedding(NDManager manager, Embedding.Builder<T> builder) {
        super(manager);
        opName = "Embedding";
        embeddingSize = builder.getEmbeddingSize();
        useDefault = builder.isUseDefault();
        dataType = builder.getDataType();
        embedding = new Parameter("embedding", this, ParameterType.WEIGHT);
        embedder = new ConcurrentHashMap<>(builder.getItems().size());
        numItems = 0;
        if (useDefault) {
            numItems++;
        }
        for (T item : builder.getItems()) {
            embedder.put(item, numItems++);
        }
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0].addAll(new Shape(embeddingSize));
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(embedding);
    }

    @Override
    public void beforeInitialize(NDList inputs) {}

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        if ("embedding".equals(name)) {
            return new Shape(numItems, embeddingSize);
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    @Override
    public NDArray forward(NDManager manager, T[][] items) {
        return forward(manager.create(embed(items)));
    }

    @Override
    public NDArray forward(NDManager manager, T[] items) {
        return forward(manager.create(embed(items)));
    }

    @Override
    public NDArray forward(NDManager manager, T item) {
        return forward(manager.create(embed(item)));
    }

    public NDArray forward(NDArray items) {
        return forward(new NDList(items)).get(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        ensureInitialized(inputs);
        NDManager manager = inputs.get(0).getManager();
        NDList result = manager.invoke(opName, opInputs(inputs), opParams(params));
        if (inputs.get(0).getShape().dimension() == 0) {
            result = new NDList(result.get(0).reshape(embeddingSize));
        }
        return result;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        embedding.save(os);
    }

    @Override
    public void loadParameters(DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        embedding.load(is);
    }

    @Override
    protected NDList opInputs(NDList inputs) {
        NDArray items = inputs.get(0);
        if (items.getShape().dimension() == 0) {
            return new NDList(items.reshape(1), embedding.getArray());
        } else {
            return new NDList(items, embedding.getArray());
        }
    }

    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("input_dim", numItems);
        result.addParam("output_dim", embeddingSize);
        result.setDataType(dataType);
        result.addParam("sparse_grad", true);
        result.addAll(params);
        return result;
    }

    private int[][] embed(T[][] items) {
        return Arrays.stream(items).map(this::embed).toArray(int[][]::new);
    }

    private int[] embed(T[] items) {
        return Arrays.stream(items).mapToInt(this::embed).toArray();
    }

    private int embed(T value) {
        if (embedder.containsKey(value)) {
            return embedder.get(value);
        } else {
            if (useDefault) {
                return 0;
            } else {
                throw new IllegalArgumentException("The provided item was not found");
            }
        }
    }
}
