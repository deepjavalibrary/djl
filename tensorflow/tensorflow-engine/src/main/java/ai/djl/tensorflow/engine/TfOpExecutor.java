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

package ai.djl.tensorflow.engine;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Preconditions;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.PointerScope;
import org.tensorflow.internal.c_api.TFE_Context;
import org.tensorflow.internal.c_api.TFE_Op;
import org.tensorflow.internal.c_api.TFE_TensorHandle;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.internal.c_api.global.tensorflow;

/** An {@code TfOpExecutor} for executing TensorFlow operation eagerly. */
final class TfOpExecutor implements AutoCloseable {

    private TfNDManager manager;
    private TFE_Op opHandle;
    private AtomicBoolean closed;

    @SuppressWarnings({"unchecked", "try"})
    TfOpExecutor(TfNDManager manager, TFE_Context eagerSessionHandle, String operation) {
        this.manager = manager;
        closed = new AtomicBoolean(false);
        try (PointerScope ignore = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            opHandle = TFE_Op.newOp(eagerSessionHandle, operation, status);
            status.throwExceptionIfNotOK();
            // keep the native pointer alive outside of the scope
            opHandle.retainReference();
        }
    }

    public NDArray[] build(int numOutputs) {
        TFE_TensorHandle[] handles = buildRawPointer(numOutputs);
        NDArray[] outputs = new NDArray[handles.length];
        for (int i = 0; i < handles.length; ++i) {
            // attach the TfNDArray along with pointer to manager
            outputs[i] = new TfNDArray(manager, handles[i]);
        }
        return outputs;
    }

    public NDArray buildSingletonOrThrow() {
        TFE_TensorHandle[] handles = buildRawPointer(1);
        try {
            Preconditions.checkArgument(
                    handles.length == 1,
                    "The expected size of outputs is 1 but got " + handles.length);
        } catch (IllegalArgumentException e) {
            Arrays.stream(handles).forEach(Pointer::close);
            throw e;
        }
        return new TfNDArray(manager, handles[0]);
    }

    // please make sure you close the output manually or attach to NDManager
    @SuppressWarnings({"unchecked", "try"})
    public TFE_TensorHandle[] buildRawPointer(int numOutputs) {
        try (PointerScope ignore = new PointerScope()) {
            IntPointer numReturnValues = new IntPointer(1).put(numOutputs);
            PointerPointer<TFE_TensorHandle> returnValues = new PointerPointer<>(numOutputs);
            TF_Status status = TF_Status.newStatus();
            // TODO(improvement): check if TFE_Execute is able to be called twice
            // and evaluate if it worth calling the TFE_Execute twice to get the # of outputs
            // in sacrifice of performance
            tensorflow.TFE_Execute(opHandle, returnValues, numReturnValues, status);
            status.throwExceptionIfNotOK();

            TFE_TensorHandle[] results = new TFE_TensorHandle[numReturnValues.get()];
            for (int i = 0; i < results.length; ++i) {
                results[i] =
                        returnValues
                                .get(TFE_TensorHandle.class, i)
                                .withDeallocator()
                                .retainReference();
            }
            return results;
        } finally {
            close();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public TfOpExecutor addInput(NDArray input) {
        try (PointerScope ignore = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            tensorflow.TFE_OpAddInput(opHandle, ((TfNDArray) input).getHandle(), status);
            status.throwExceptionIfNotOK();
        }
        return this;
    }

    @SuppressWarnings({"unchecked", "try"})
    public TfOpExecutor addInputList(NDArray[] inputs) {
        TFE_TensorHandle[] inputHandles =
                Arrays.stream(inputs)
                        .map(array -> ((TfNDArray) array).getHandle())
                        .toArray(TFE_TensorHandle[]::new);
        try (PointerScope ignore = new PointerScope()) {
            PointerPointer<TFE_TensorHandle> tensorPointers =
                    new PointerPointer<>(inputHandles.length);
            for (int i = 0; i < inputHandles.length; ++i) {
                tensorPointers.put(i, inputHandles[i]);
            }
            TF_Status status = TF_Status.newStatus();
            tensorflow.TFE_OpAddInputList(opHandle, tensorPointers, inputHandles.length, status);
            status.throwExceptionIfNotOK();
        }
        return this;
    }

    @SuppressWarnings({"unchecked", "try"})
    public TfOpExecutor setDevice(Device device) {
        String deviceStr;
        try (PointerScope ignore = new PointerScope()) {
            if (device.getDeviceType().equals(Device.Type.CPU)) {
                deviceStr = "/device:CPU:0";
            } else if (device.getDeviceType().equals(Device.Type.GPU)) {
                deviceStr = "/device:GPU:" + device.getDeviceId();
            } else {
                throw new EngineException("Unknown device type to TensorFlow Engine: " + device);
            }
            TF_Status status = TF_Status.newStatus();
            tensorflow.TFE_OpSetDevice(opHandle, deviceStr, status);
            status.throwExceptionIfNotOK();
            return this;
        } catch (Exception e) {
            close();
            throw e;
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public TfOpExecutor addParam(String name, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        try (PointerScope ignore = new PointerScope()) {
            tensorflow.TFE_OpSetAttrString(opHandle, name, new BytePointer(bytes), bytes.length);
        }
        return this;
    }

    public TfOpExecutor addParam(String name, long value) {
        tensorflow.TFE_OpSetAttrInt(opHandle, name, value);
        return this;
    }

    public TfOpExecutor addParam(String name, float value) {
        tensorflow.TFE_OpSetAttrFloat(opHandle, name, value);
        return this;
    }

    public TfOpExecutor addParam(String name, boolean value) {
        tensorflow.TFE_OpSetAttrBool(opHandle, name, (byte) (value ? 1 : 0));
        return this;
    }

    public TfOpExecutor addParam(String name, DataType dataType) {
        tensorflow.TFE_OpSetAttrType(opHandle, name, TfDataType.toTf(dataType));
        return this;
    }

    public TfOpExecutor addParam(String name, long[] values) {
        tensorflow.TFE_OpSetAttrIntList(opHandle, name, values, values.length);
        return this;
    }

    @Override
    public void close() {
        if (closed.getAndSet(true) || opHandle == null || opHandle.isNull()) {
            return;
        }
        opHandle.close();
    }
}
