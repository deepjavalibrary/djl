/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.ndarray.NDList;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

/** {@code RsSymbolBlock} is the Rust implementation of {@link SymbolBlock}. */
public class RsSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private AtomicReference<Long> handle;
    private String uid;
    private RsNDManager manager;

    /**
     * Constructs a {@code RsSymbolBlock}.
     *
     * <p>You can create a {@code RsSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param manager the manager to use for the block
     * @param handle the Booster handle
     */
    @SuppressWarnings("this-escape")
    public RsSymbolBlock(RsNDManager manager, long handle) {
        this.handle = new AtomicReference<>(handle);
        this.manager = manager;
        inputNames = Arrays.asList(RustLibrary.getInputNames(handle));
        uid = String.valueOf(handle);
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        if (inputNames.size() != inputs.size()) {
            throw new IllegalArgumentException("Input size mismatch, requires: " + inputNames);
        }
        try (RsNDManager sub = (RsNDManager) manager.newSubManager()) {
            long[] inputHandles = new long[inputs.size()];
            for (int i = 0; i < inputs.size(); i++) {
                inputHandles[i] = sub.from(inputs.get(i)).getHandle();
            }
            long outputHandle = RustLibrary.runInference(handle.get(), inputHandles);
            RsNDArray output = new RsNDArray(manager, outputHandle);
            output.attach(inputs.head().getManager());
            return new NDList(output);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            manager.detachInternal(uid);
            manager = null;
        }
    }

    /**
     * Gets the native Rust pointer.
     *
     * @return the pointer
     */
    public Long getHandle() {
        Long reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("Rust model handle has been released!");
        }
        return reference;
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("Not yet supported");
    }
}
