/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.tensorrt.engine;

import ai.djl.ndarray.NDList;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.tensorrt.jni.JniUtils;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.concurrent.atomic.AtomicReference;

/**
 * {@code TrtSymbolBlock} is the TensorRT implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code TrtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class TrtSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private AtomicReference<Long> handle;

    /**
     * Constructs a {@code TrtSymbolBlock}.
     *
     * <p>You can create a {@code TrtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param handle the handle for native TensorRT model
     */
    public TrtSymbolBlock(long handle) {
        this.handle = new AtomicReference<>(handle);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        throw new UnsupportedOperationException("Use TrtExecutionContext instead.");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteTrtModel(pointer);
        }
    }

    TrtSession createSession(TrtNDManager manager) {
        long session = JniUtils.createSession(handle.get());
        return new TrtSession(manager, handle.get(), session);
    }
}
