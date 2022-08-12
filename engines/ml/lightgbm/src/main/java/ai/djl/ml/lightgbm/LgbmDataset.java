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
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/** A special {@link NDArray} used by LightGBM for training models. */
public class LgbmDataset extends NDArrayAdapter {

    private AtomicReference<SWIGTYPE_p_p_void> handle;

    // Track Dataset source for inference calls
    private SrcType srcType;
    private Path srcFile;
    private NDArray srcArray;

    LgbmDataset(NDManager manager, NDManager alternativeManager, LgbmNDArray array) {
        super(
                manager,
                alternativeManager,
                array.getShape(),
                array.getDataType(),
                UUID.randomUUID().toString());
        srcType = SrcType.ARRAY;
        srcArray = array;
        handle = new AtomicReference<>();
    }

    LgbmDataset(NDManager manager, NDManager alternativeManager, Path file) {
        super(manager, alternativeManager, null, DataType.FLOAT32, UUID.randomUUID().toString());
        srcType = SrcType.FILE;
        srcFile = file;
        handle = new AtomicReference<>();
    }

    /**
     * Gets the native LightGBM Dataset pointer.
     *
     * @return the pointer
     */
    public SWIGTYPE_p_p_void getHandle() {
        SWIGTYPE_p_p_void pointer = handle.get();
        if (pointer == null) {
            synchronized (this) {
                switch (getSrcType()) {
                    case FILE:
                        handle.set(JniUtils.datasetFromFile(getSrcFile().toString()));
                        break;
                    case ARRAY:
                        handle.set(JniUtils.datasetFromArray(getSrcArrayConverted()));
                        break;
                    default:
                        throw new IllegalArgumentException("Unexpected SrcType");
                }
            }
        }
        return pointer;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape =
                    new Shape(
                            JniUtils.datasetGetRows(handle.get()),
                            JniUtils.datasetGetCols(handle.get()));
        }
        return shape;
    }

    /**
     * Returns the type of source data for the {@link LgbmDataset}.
     *
     * @return the type of source data for the {@link LgbmDataset}
     */
    public SrcType getSrcType() {
        return srcType;
    }

    /**
     * Returns the file used to create this (if applicable).
     *
     * @return the file used to create this (if applicable)
     */
    public Path getSrcFile() {
        return srcFile;
    }

    /**
     * Returns the array used to create this (if applicable).
     *
     * @return the array used to create this (if applicable)
     */
    public NDArray getSrcArray() {
        return srcArray;
    }

    /**
     * Returns the array used to create this (if applicable) converted into an {@link LgbmNDArray}.
     *
     * @return the array used to create this (if applicable) converted into an {@link LgbmNDArray}
     */
    public LgbmNDArray getSrcArrayConverted() {
        NDArray a = getSrcArray();
        if (a instanceof LgbmNDArray) {
            return (LgbmNDArray) a;
        } else {
            return new LgbmNDArray(
                    manager, alternativeManager, a.toByteBuffer(), a.getShape(), a.getDataType());
        }
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        throw new UnsupportedOperationException("Not supported by the LgbmDataset yet");
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = LgbmNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        throw new UnsupportedOperationException("Not supported by the LgbmDataset yet");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        SWIGTYPE_p_p_void pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.freeDataset(pointer);
        }
    }

    /** The type of data used to create the {@link LgbmDataset}. */
    public enum SrcType {
        FILE,
        ARRAY
    }
}
