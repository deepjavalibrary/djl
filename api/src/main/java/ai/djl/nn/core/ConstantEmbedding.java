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
package ai.djl.nn.core;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Optional;

/** An {@link AbstractIndexedEmbedding} that always returns a constant value. */
@SuppressWarnings("rawtypes")
public class ConstantEmbedding extends AbstractBlock implements AbstractIndexedEmbedding {

    protected NDArray embedding;

    /**
     * Constructs a constant embedding with the given constant.
     *
     * @param embedding the value to return for all embeddings
     */
    public ConstantEmbedding(NDArray embedding) {
        this.embedding = embedding;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDManager manager = inputs.get(0).getManager();
        NDArray base = manager.create(embedding.getShape());
        embedding.copyTo(base);
        Shape shape = inputs.get(0).getShape().addAll(embedding.getShape());
        return new NDList(base.repeat(shape));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {inputShapes[0].addAll(embedding.getShape())};
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) {
        // Nothing to save
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) {
        // Nothing to load
    }

    /** {@inheritDoc} */
    @Override
    public Optional<?> unembed(long index) {
        return Optional.empty();
    }

    /** {@inheritDoc} */
    @Override
    public byte[] encode(Object input) {
        return new byte[0];
    }

    /** {@inheritDoc} */
    @Override
    public Object decode(byte[] byteArray) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long embed(Object item) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embed(NDManager manager, Object[] items) {
        NDArray base = manager.create(embedding.getShape());
        embedding.copyTo(base);
        Shape shape = new Shape(items.length).addAll(embedding.getShape());
        return base.repeat(shape);
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasItem(Object item) {
        return true;
    }
}
