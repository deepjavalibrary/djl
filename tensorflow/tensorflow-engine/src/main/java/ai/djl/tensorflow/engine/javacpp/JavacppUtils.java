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

package ai.djl.tensorflow.engine.javacpp;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.tensorflow.engine.SavedModelBundle;
import ai.djl.tensorflow.engine.TfDataType;
import ai.djl.util.Pair;
import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.PointerScope;
import org.tensorflow.exceptions.TensorFlowException;
import org.tensorflow.internal.c_api.AbstractTFE_Context;
import org.tensorflow.internal.c_api.AbstractTFE_TensorHandle;
import org.tensorflow.internal.c_api.AbstractTF_Graph;
import org.tensorflow.internal.c_api.AbstractTF_Tensor;
import org.tensorflow.internal.c_api.TFE_Context;
import org.tensorflow.internal.c_api.TFE_ContextOptions;
import org.tensorflow.internal.c_api.TFE_TensorHandle;
import org.tensorflow.internal.c_api.TF_Buffer;
import org.tensorflow.internal.c_api.TF_Graph;
import org.tensorflow.internal.c_api.TF_Operation;
import org.tensorflow.internal.c_api.TF_Output;
import org.tensorflow.internal.c_api.TF_Session;
import org.tensorflow.internal.c_api.TF_SessionOptions;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.internal.c_api.TF_TString;
import org.tensorflow.internal.c_api.TF_Tensor;
import org.tensorflow.internal.c_api.global.tensorflow;
import org.tensorflow.proto.framework.ConfigProto;
import org.tensorflow.proto.framework.GPUOptions;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.RunOptions;

/** A class containing utilities to interact with the TensorFlow Engine's Javacpp layer. */
public final class JavacppUtils {

    private static final Pattern DEVICE_PATTERN = Pattern.compile(".*device:([A-Z]PU):(\\d+)");

    private JavacppUtils() {}

    @SuppressWarnings({"unchecked", "try"})
    public static SavedModelBundle loadSavedModelBundle(
            String exportDir, String[] tags, ConfigProto config, RunOptions runOptions) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();

            // allocate parameters for TF_LoadSessionFromSavedModel
            TF_SessionOptions opts = TF_SessionOptions.newSessionOptions();
            if (config != null) {
                BytePointer configBytes = new BytePointer(config.toByteArray());
                tensorflow.TF_SetConfig(opts, configBytes, configBytes.capacity(), status);
                status.throwExceptionIfNotOK();
            }
            TF_Buffer runOpts = TF_Buffer.newBufferFromString(runOptions);

            // load the session
            TF_Graph graphHandle = AbstractTF_Graph.newGraph().retainReference();
            TF_Buffer metaGraphDef = TF_Buffer.newBuffer();
            TF_Session sessionHandle =
                    tensorflow.TF_LoadSessionFromSavedModel(
                            opts,
                            runOpts,
                            new BytePointer(exportDir),
                            new PointerPointer<>(tags),
                            tags.length,
                            graphHandle,
                            metaGraphDef,
                            status);
            status.throwExceptionIfNotOK();

            // handle the result
            try {
                return new SavedModelBundle(
                        graphHandle,
                        sessionHandle,
                        MetaGraphDef.parseFrom(metaGraphDef.dataAsByteBuffer()));
            } catch (InvalidProtocolBufferException e) {
                throw new TensorFlowException("Cannot parse MetaGraphDef protocol buffer", e);
            }
        }
    }

    private static TF_Operation getGraphOpByName(TF_Graph graphHandle, String operation) {
        TF_Operation opHandle;
        synchronized (graphHandle) {
            opHandle = tensorflow.TF_GraphOperationByName(graphHandle, operation);
        }
        if (opHandle == null || opHandle.isNull()) {
            throw new IllegalArgumentException(
                    "No Operation named [" + operation + "] in the Graph");
        }
        return opHandle;
    }

    public static Pair<TF_Operation, Integer> getGraphOperationByName(
            TF_Graph graphHandle, String operation) {
        int colon = operation.lastIndexOf(':');
        if (colon == -1 || colon == operation.length() - 1) {
            return new Pair<>(getGraphOpByName(graphHandle, operation), 0);
        }
        try {
            String op = operation.substring(0, colon);
            int index = Integer.parseInt(operation.substring(colon + 1));
            return new Pair<>(getGraphOpByName(graphHandle, op), index);
        } catch (NumberFormatException e) {
            return new Pair<>(getGraphOpByName(graphHandle, operation), 0);
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TF_Tensor[] runSession(
            TF_Session handle,
            RunOptions runOptions,
            TF_Tensor[] inputTensorHandles,
            TF_Operation[] inputOpHandles,
            int[] inputOpIndices,
            TF_Operation[] outputOpHandles,
            int[] outputOpIndices,
            TF_Operation[] targetOpHandles) {
        int numInputs = inputTensorHandles.length;
        int numOutputs = outputOpHandles.length;
        int numTargets = targetOpHandles.length;
        try (PointerScope ignored = new PointerScope()) {
            // TODO: check with sig-jvm if TF_Output here is freed
            TF_Output inputs = new TF_Output(numInputs);
            PointerPointer<TF_Tensor> inputValues = new PointerPointer<>(numInputs);
            TF_Output outputs = new TF_Output(numOutputs);
            PointerPointer<TF_Tensor> outputValues = new PointerPointer<>(numOutputs);
            PointerPointer<TF_Operation> targets = new PointerPointer<>(numTargets);

            // set input
            for (int i = 0; i < numInputs; ++i) {
                inputValues.put(i, inputTensorHandles[i]);
            }

            // set TF_Output for inputs
            for (int i = 0; i < numInputs; ++i) {
                inputs.position(i).oper(inputOpHandles[i]).index(inputOpIndices[i]);
            }
            inputs.position(0);

            // set TF_Output for outputs
            for (int i = 0; i < numOutputs; ++i) {
                outputs.position(i).oper(outputOpHandles[i]).index(outputOpIndices[i]);
            }
            outputs.position(0);

            // set target
            for (int i = 0; i < numTargets; ++i) {
                targets.put(i, targetOpHandles[i]);
            }
            TF_Status status = TF_Status.newStatus();
            TF_Buffer runOpts = TF_Buffer.newBufferFromString(runOptions);

            tensorflow.TF_SessionRun(
                    handle,
                    runOpts,
                    inputs,
                    inputValues,
                    numInputs,
                    outputs,
                    outputValues,
                    numOutputs,
                    targets,
                    numTargets,
                    null,
                    status);
            status.throwExceptionIfNotOK();

            TF_Tensor[] ret = new TF_Tensor[numOutputs];
            for (int i = 0; i < numOutputs; ++i) {
                ret[i] = outputValues.get(TF_Tensor.class, i).withDeallocator().retainReference();
            }
            return ret;
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TFE_Context createEagerSession(
            boolean async, int devicePlacementPolicy, ConfigProto config) {
        try (PointerScope ignored = new PointerScope()) {
            TFE_ContextOptions opts = TFE_ContextOptions.newContextOptions();
            TF_Status status = TF_Status.newStatus();
            if (config != null) {
                BytePointer configBytes = new BytePointer(config.toByteArray());
                tensorflow.TFE_ContextOptionsSetConfig(
                        opts, configBytes, configBytes.capacity(), status);
                status.throwExceptionIfNotOK();
            }
            tensorflow.TFE_ContextOptionsSetAsync(opts, (byte) (async ? 1 : 0));
            tensorflow.TFE_ContextOptionsSetDevicePlacementPolicy(opts, devicePlacementPolicy);
            TFE_Context context = AbstractTFE_Context.newContext(opts, status);
            status.throwExceptionIfNotOK();
            return context.retainReference();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static Device getDevice(TFE_TensorHandle handle) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            BytePointer pointer = tensorflow.TFE_TensorHandleDeviceName(handle, status);
            String device = new String(pointer.getStringBytes(), StandardCharsets.UTF_8);
            return fromTfDevice(device);
        }
    }

    public static DataType getDataType(TFE_TensorHandle handle) {
        return TfDataType.fromTf(tensorflow.TFE_TensorHandleDataType(handle));
    }

    @SuppressWarnings({"unchecked", "try"})
    public static Shape getShape(TFE_TensorHandle handle) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            int numDims = tensorflow.TFE_TensorHandleNumDims(handle, status);
            status.throwExceptionIfNotOK();
            long[] shapeArr = new long[numDims];
            for (int i = 0; i < numDims; i++) {
                shapeArr[i] = tensorflow.TFE_TensorHandleDim(handle, i, status);
                status.throwExceptionIfNotOK();
            }
            return new Shape(shapeArr);
        }
    }

    public static TF_Tensor createEmptyTFTensor(Shape shape, DataType dataType) {
        int dType = TfDataType.toTf(dataType);
        long[] dims = shape.getShape();
        long numBytes = dataType.getNumOfBytes() * shape.size();
        TF_Tensor tensor = AbstractTF_Tensor.allocateTensor(dType, dims, numBytes);
        if (tensor == null || tensor.isNull()) {
            throw new IllegalStateException("unable to allocate memory for the Tensor");
        }
        return tensor;
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TFE_TensorHandle createEmptyTFETensor(Shape shape, DataType dataType) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Tensor tensor = createEmptyTFTensor(shape, dataType);
            TF_Status status = TF_Status.newStatus();
            TFE_TensorHandle handle = AbstractTFE_TensorHandle.newTensor(tensor, status);
            status.throwExceptionIfNotOK();
            return handle.retainReference();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static Pair<TF_Tensor, TFE_TensorHandle> createStringTensor(String src) {
        int dType = TfDataType.toTf(DataType.STRING);
        long[] dims = {};
        long numBytes = Loader.sizeof(TF_TString.class);
        try (PointerScope ignored = new PointerScope()) {
            /*
             * String tensor allocates a separate TF_TString memory. The TF_TString will
             * be deleted when the string tensor is closed. We have to track TF_TString
             * memory by ourselves and make sure thw TF_TString lifecycle align with
             * TFE_TensorHandle. TF_Tensor already handles TF_TString automatically, We
             * can just keep a TF_Tensor reference in TfNDArray.
             */
            TF_Tensor tensor = AbstractTF_Tensor.allocateTensor(dType, dims, numBytes);
            Pointer pointer = tensorflow.TF_TensorData(tensor).capacity(numBytes);
            TF_TString data = new TF_TString(pointer).capacity(pointer.position() + 1);
            byte[] buf = src.getBytes(StandardCharsets.UTF_8);
            tensorflow.TF_TString_Copy(data, new BytePointer(buf), buf.length);
            TF_Status status = TF_Status.newStatus();
            TFE_TensorHandle handle = AbstractTFE_TensorHandle.newTensor(tensor, status);
            status.throwExceptionIfNotOK();

            handle.retainReference();
            tensor.retainReference();
            return new Pair<>(tensor, handle);
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TFE_TensorHandle createTFETensorFromByteBuffer(
            ByteBuffer buf, Shape shape, DataType dataType) {
        int dType = TfDataType.toTf(dataType);
        long[] dims = shape.getShape();
        long numBytes;
        if (dataType == DataType.STRING) {
            numBytes = buf.remaining() + 1;
        } else {
            numBytes = shape.size() * dataType.getNumOfBytes();
        }
        try (PointerScope ignored = new PointerScope()) {
            TF_Tensor tensor = AbstractTF_Tensor.allocateTensor(dType, dims, numBytes);
            // get data pointer in native engine
            Pointer pointer = tensorflow.TF_TensorData(tensor).capacity(numBytes);
            pointer.asByteBuffer().put(buf);
            TF_Status status = TF_Status.newStatus();
            TFE_TensorHandle handle = AbstractTFE_TensorHandle.newTensor(tensor, status);
            status.throwExceptionIfNotOK();
            return handle.retainReference();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TF_Tensor resolveTFETensor(TFE_TensorHandle handle) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            TF_Tensor tensor = tensorflow.TFE_TensorHandleResolve(handle, status).withDeallocator();
            status.throwExceptionIfNotOK();
            return tensor.retainReference();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TFE_TensorHandle createTFETensor(TF_Tensor handle) {
        try (PointerScope ignored = new PointerScope()) {
            TF_Status status = TF_Status.newStatus();
            TFE_TensorHandle tensor = AbstractTFE_TensorHandle.newTensor(handle, status);
            status.throwExceptionIfNotOK();
            return tensor.retainReference();
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static String[] getString(TFE_TensorHandle handle, int count, Charset charset) {
        try (PointerScope ignored = new PointerScope()) {
            // convert to TF_Tensor
            TF_Status status = TF_Status.newStatus();
            // should not add .withDeallocator() here, otherwise sting data will be destroyed
            TF_Tensor tensor = tensorflow.TFE_TensorHandleResolve(handle, status);
            status.throwExceptionIfNotOK();

            long tensorSize = tensorflow.TF_TensorByteSize(tensor);
            Pointer pointer = tensorflow.TF_TensorData(tensor).capacity(tensorSize);

            TF_TString data = new TF_TString(pointer).capacity(pointer.position() + count);
            String[] ret = new String[count];
            for (int i = 0; i < count; ++i) {
                TF_TString tstring = data.getPointer(i);
                long size = tensorflow.TF_TString_GetSize(tstring);
                BytePointer bp = tensorflow.TF_TString_GetDataPointer(tstring).capacity(size);
                ret[i] = bp.getString(charset);
            }

            tensorflow.TF_DeleteTensor(tensor); // manually delete tensor
            return ret;
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static ByteBuffer getByteBuffer(TFE_TensorHandle handle) {
        try (PointerScope ignored = new PointerScope()) {
            // convert to TF_Tensor
            TF_Status status = TF_Status.newStatus();
            TF_Tensor tensor = tensorflow.TFE_TensorHandleResolve(handle, status).withDeallocator();
            status.throwExceptionIfNotOK();
            Pointer pointer =
                    tensorflow.TF_TensorData(tensor).capacity(tensorflow.TF_TensorByteSize(tensor));
            ByteBuffer buf = pointer.asByteBuffer();
            // do the copy since we should make sure the returned ByteBuffer is still valid after
            // the tensor is deleted
            ByteBuffer ret = ByteBuffer.allocate(buf.capacity());
            buf.rewind();
            ret.put(buf);
            ret.flip();
            return ret.order(ByteOrder.nativeOrder());
        }
    }

    @SuppressWarnings({"unchecked", "try"})
    public static TFE_TensorHandle toDevice(
            TFE_TensorHandle handle, TFE_Context eagerSessionHandle, Device device) {
        try (PointerScope ignored = new PointerScope()) {
            String deviceName = toTfDevice(device);
            TF_Status status = TF_Status.newStatus();
            TFE_TensorHandle newHandle =
                    tensorflow.TFE_TensorHandleCopyToDevice(
                            handle, eagerSessionHandle, deviceName, status);
            status.throwExceptionIfNotOK();
            return newHandle;
        }
    }

    public static ConfigProto getSessionConfig() {
        Integer interop = Integer.getInteger("ai.djl.tensorflow.num_interop_threads");
        Integer intraop = Integer.getInteger("ai.djl.tensorflow.num_intraop_threads");
        ConfigProto.Builder configBuilder = ConfigProto.newBuilder();
        if (interop != null) {
            configBuilder.setInterOpParallelismThreads(interop);
        }
        if (intraop != null) {
            configBuilder.setIntraOpParallelismThreads(intraop);
        }
        GPUOptions gpuOptions = GPUOptions.newBuilder().setVisibleDeviceList("0").build();
        configBuilder.setGpuOptions(gpuOptions);
        return configBuilder.build();
    }

    public static Device fromTfDevice(String device) {
        Matcher m = DEVICE_PATTERN.matcher(device);
        if (m.matches()) {
            if ("CPU".equals(m.group(1))) {
                return Device.cpu();
            } else if ("GPU".equals(m.group(1))) {
                return Device.of(Device.Type.GPU, Integer.parseInt(m.group(2)));
            }
        }
        throw new EngineException("Unknown device type to TensorFlow Engine: " + device);
    }

    public static String toTfDevice(Device device) {
        if (device.getDeviceType().equals(Device.Type.CPU)) {
            return "/device:CPU:0";
        } else if (device.getDeviceType().equals(Device.Type.GPU)) {
            return "/device:GPU:" + device.getDeviceId();
        } else {
            throw new EngineException("Unknown device type to TensorFlow Engine: " + device);
        }
    }
}
