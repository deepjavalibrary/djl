/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ml.lightgbm;

import ai.djl.ml.lightgbm.jni.JniUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;

/** {@code LgbmSymbolBlock} is the LightGBM implementation of {@link SymbolBlock}. */
public class LgbmSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private AtomicReference<SWIGTYPE_p_p_void> handle;
    private int iterations;
    private String uid;
    private LgbmNDManager manager;
    private int inferenceType;

    /**
     * Constructs a {@code LgbmSymbolBlock}.
     *
     * <p>You can create a {@code LgbmSymbolBlock} using {@link
     * ai.djl.Model#load(java.nio.file.Path, String)}.
     *
     * @param manager the manager to use for the block
     * @param iterations the number of iterations the model was trained for
     * @param handle the Booster handle
     */
    @SuppressWarnings("this-escape")
    public LgbmSymbolBlock(LgbmNDManager manager, int iterations, SWIGTYPE_p_p_void handle) {
        this.handle = new AtomicReference<>(handle);
        this.iterations = iterations;
        this.manager = manager;
        uid = String.valueOf(handle);
        manager.attachInternal(uid, this);
        this.inferenceType = lightgbmlibConstants.C_API_PREDICT_NORMAL;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray array = inputs.singletonOrThrow();
        try (LgbmNDManager sub = (LgbmNDManager) manager.newSubManager()) {
            LgbmNDArray lgbmNDArray = sub.from(array);
            Pair<Integer, ByteBuffer> result =
                    JniUtils.inference(getHandle(), iterations, lgbmNDArray, inferenceType);

            NDArray ret =
                    manager.create(
                            result.getValue(),
                            new Shape(result.getKey()),
                            lgbmNDArray.getDataType());
            ret.attach(array.getManager());
            return new NDList(ret);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        SWIGTYPE_p_p_void pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.freeModel(pointer);
            manager.detachInternal(uid);
            manager = null;
        }
    }

    /**
     * Gets the native LightGBM Booster pointer.
     *
     * @return the pointer
     */
    public SWIGTYPE_p_p_void getHandle() {
        SWIGTYPE_p_p_void pointer = handle.get();
        if (pointer == null) {
            throw new IllegalStateException("LightGBM model handle has been released!");
        }
        return pointer;
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("Not yet supported");
    }

    void setInferenceType(String inferenceType) {
        switch (inferenceType) {
            case "NORMAL":
                this.inferenceType = lightgbmlibConstants.C_API_PREDICT_NORMAL;
                break;
            case "RAW_SCORE":
                this.inferenceType = lightgbmlibConstants.C_API_PREDICT_RAW_SCORE;
                break;
            case "LEAF_INDEX":
                this.inferenceType = lightgbmlibConstants.C_API_PREDICT_LEAF_INDEX;
                break;
            case "CONTRIB":
                this.inferenceType = lightgbmlibConstants.C_API_PREDICT_CONTRIB;
                break;
            default:
                throw new IllegalArgumentException(
                        "Unsupported inference type: "
                                + inferenceType
                                + ". Supported types include: NORMAL, RAW_SCORE, LEAF_INDEX,"
                                + " CONTRIB.");
        }
    }

    String getInferenceType() {
        if (inferenceType == lightgbmlibConstants.C_API_PREDICT_NORMAL) {
            return "NORMAL";
        } else if (inferenceType == lightgbmlibConstants.C_API_PREDICT_RAW_SCORE) {
            return "RAW_SCORE";
        } else if (inferenceType == lightgbmlibConstants.C_API_PREDICT_LEAF_INDEX) {
            return "LEAF_INDEX";
        } else if (inferenceType == lightgbmlibConstants.C_API_PREDICT_CONTRIB) {
            return "CONTRIB";
        } else {
            throw new AssertionError("Unexpected inference type: " + inferenceType);
        }
    }
}
