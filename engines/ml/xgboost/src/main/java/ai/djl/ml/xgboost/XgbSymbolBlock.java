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
package ai.djl.ml.xgboost;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;
import ml.dmlc.xgboost4j.java.JniUtils;

/** {@code XgbSymbolBlock} is the XGBoost implementation of {@link SymbolBlock}. */
public class XgbSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private AtomicReference<Long> handle;
    private String uid;
    private XgbNDManager manager;
    private Mode mode;
    private int treeLimit;

    /**
     * Constructs a {@code XgbSymbolBlock}.
     *
     * <p>You can create a {@code XgbSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param manager the manager to use for the block
     * @param handle the Booster handle
     */
    public XgbSymbolBlock(XgbNDManager manager, long handle) {
        this.handle = new AtomicReference<>(handle);
        this.manager = manager;
        uid = String.valueOf(handle);
        manager.attachInternal(uid, this);
        mode = Mode.DEFAULT;
        treeLimit = 0;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray array = inputs.singletonOrThrow();
        try (XgbNDManager sub = (XgbNDManager) manager.newSubManager()) {
            XgbNDArray xgbNDArray = sub.from(array);
            // TODO: return DirectBuffer from JNI to avoid copy
            float[] result = JniUtils.inference(this, xgbNDArray, treeLimit, mode);
            ByteBuffer buf = manager.allocateDirect(result.length * 4);
            buf.asFloatBuffer().put(result);
            buf.rewind();

            NDArray ret = manager.create(buf, new Shape(result.length), DataType.FLOAT32);
            ret.attach(array.getManager());
            return new NDList(ret);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteModel(pointer);
            manager.detachInternal(uid);
            manager = null;
        }
    }

    /**
     * Gets the native XGBoost Booster pointer.
     *
     * @return the pointer
     */
    public Long getHandle() {
        Long reference = handle.get();
        if (reference == null) {
            throw new IllegalStateException("XGBoost model handle has been released!");
        }
        return reference;
    }

    void setMode(Mode mode) {
        this.mode = mode;
    }

    void setTreeLimit(int treeLimit) {
        this.treeLimit = treeLimit;
    }

    /** The mode of inference for OptionMask. */
    public enum Mode {
        DEFAULT(0),
        OUTPUT_MARGIN(1),
        LEAF(2),
        CONTRIB(4);

        private int value;

        Mode(int value) {
            this.value = value;
        }

        /**
         * Gets the value of the mode.
         *
         * @return the value in number
         */
        public int getValue() {
            return value;
        }
    }
}
