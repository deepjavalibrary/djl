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
package ai.djl.mxnet.jna;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.mxnet.engine.CachedOp;
import ai.djl.mxnet.engine.MxDeviceType;
import ai.djl.mxnet.engine.MxNDArray;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.engine.MxSymbolBlock;
import ai.djl.mxnet.engine.Symbol;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A class containing utilities to interact with the MXNet Engine's Java Native Access (JNA) layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JnaUtils {

    public static final String[] EMPTY_ARRAY = new String[0];
    public static final ObjectPool<PointerByReference> REFS =
            new ObjectPool<>(PointerByReference::new, r -> r.setValue(null));

    /** An enum that enumerates the statuses of numpy mode. */
    public enum NumpyMode {
        OFF,
        THREAD_LOCAL_ON,
        GLOBAL_ON
    }

    private static final String[] OP_NAME_PREFIX = {
        "_contrib_", "_linalg_", "_sparse_", "_image_", "_random_"
    };

    private static final MxnetLibrary LIB = LibUtils.loadLibrary();

    private static final Map<String, FunctionInfo> OPS = getNdArrayFunctions();
    private static final Set<String> FEATURES = getFeaturesInternal();

    private JnaUtils() {}

    /////////////////////////////////
    // MXNet information
    /////////////////////////////////

    public static int getVersion() {
        IntBuffer version = IntBuffer.allocate(1);
        checkCall(LIB.MXGetVersion(version));

        return version.get();
    }

    public static Set<String> getAllOpNames() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference outArray = REFS.acquire();

        checkCall(LIB.MXListAllOpNames(outSize, outArray));

        int size = outSize.get();
        Pointer[] pointers = outArray.getValue().getPointerArray(0, size);

        Set<String> set = new HashSet<>();
        for (Pointer p : pointers) {
            set.add(p.getString(0, StandardCharsets.UTF_8.name()));
        }
        REFS.recycle(outArray);
        return set;
    }

    public static Map<String, FunctionInfo> getNdArrayFunctions() {
        Set<String> opNames = JnaUtils.getAllOpNames();
        Map<String, FunctionInfo> map = new ConcurrentHashMap<>();

        PointerByReference ref = REFS.acquire();
        for (String opName : opNames) {
            checkCall(LIB.NNGetOpHandle(opName, ref));

            String functionName = getOpNamePrefix(opName);

            // System.out.println("Name: " + opName + "/" + functionName);
            map.put(functionName, getFunctionByName(opName, functionName, ref.getValue()));
            ref.setValue(null);
        }
        REFS.recycle(ref);
        return map;
    }

    public static FunctionInfo op(String opName) {
        if (!OPS.containsKey(opName)) {
            throw new IllegalArgumentException("Unknown operator: " + opName);
        }
        return OPS.get(opName);
    }

    private static FunctionInfo getFunctionByName(
            String name, String functionName, Pointer handle) {
        String[] nameRef = {name};
        String[] description = new String[1];
        IntBuffer numArgs = IntBuffer.allocate(1);
        PointerByReference argNameRef = REFS.acquire();
        PointerByReference argTypeRef = REFS.acquire();
        PointerByReference argDescRef = REFS.acquire();
        String[] keyVarArgs = new String[1];
        String[] returnType = new String[1];

        checkCall(
                LIB.MXSymbolGetAtomicSymbolInfo(
                        handle,
                        nameRef,
                        description,
                        numArgs,
                        argNameRef,
                        argTypeRef,
                        argDescRef,
                        keyVarArgs,
                        returnType));

        int count = numArgs.get();
        PairList<String, String> arguments = new PairList<>();
        if (count != 0) {
            String[] argNames =
                    argNameRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            String[] argTypes =
                    argTypeRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            for (int i = 0; i < argNames.length; i++) {
                arguments.add(argNames[i], argTypes[i]);
            }
        }

        REFS.recycle(argNameRef);
        REFS.recycle(argTypeRef);
        REFS.recycle(argDescRef);

        return new FunctionInfo(handle, functionName, arguments);
    }

    /*
    int MXFuncGetInfo(Pointer fun, String name[], String description[], IntBuffer num_args,
                      PointerByReference arg_names, PointerByReference arg_type_infos,
                      PointerByReference arg_descriptions, String return_type[]);

    int MXFuncDescribe(Pointer fun, IntBuffer num_use_vars, IntBuffer num_scalars,
                       IntBuffer num_mutate_vars, IntBuffer type_mask);

    int MXFuncInvoke(Pointer fun, PointerByReference use_vars, FloatBuffer scalar_args,
                     PointerByReference mutate_vars);

    int MXFuncInvokeEx(Pointer fun, PointerByReference use_vars, FloatBuffer scalar_args,
                       PointerByReference mutate_vars, int num_params,
                       PointerByReference param_keys, PointerByReference param_vals);
    */

    /////////////////////////////////
    // System information
    /////////////////////////////////

    public static int getGpuCount() {
        IntBuffer count = IntBuffer.allocate(1);
        checkCall(LIB.MXGetGPUCount(count));

        return count.get();
    }

    public static long[] getGpuMemory(Device device) {
        if (!device.isGpu()) {
            throw new IllegalArgumentException("Only GPU device is allowed.");
        }

        int deviceId = device.getDeviceId();
        long[] ret = new long[2];

        LongBuffer freeMem = LongBuffer.wrap(ret, 0, 1);
        LongBuffer totalMem = LongBuffer.wrap(ret, 1, 1);

        checkCall(LIB.MXGetGPUMemoryInformation64(deviceId, freeMem, totalMem));

        return ret;
    }

    /* Need tests
    public static void setOmpThreads(int threads) {
        checkCall(LIB.MXSetNumOMPThreads(threads));
    }

    public static int setBulkSize(int bulkSize) {
        IntBuffer prevBulkSize = IntBuffer.allocate(1);
        checkCall(LIB.MXEngineSetBulkSize(bulkSize, prevBulkSize));

        return prevBulkSize.get();
    }
    */

    /////////////////////////////////
    // Utilities
    /////////////////////////////////

    public static Set<String> getFeatures() {
        return FEATURES;
    }

    private static Set<String> getFeaturesInternal() {
        PointerByReference ref = REFS.acquire();
        NativeSizeByReference outSize = new NativeSizeByReference();
        checkCall(LIB.MXLibInfoFeatures(ref, outSize));

        int size = outSize.getValue().intValue();
        if (size == 0) {
            REFS.recycle(ref);
            return Collections.emptySet();
        }

        LibFeature pointer = new LibFeature(ref.getValue());
        pointer.read();

        LibFeature[] features = (LibFeature[]) pointer.toArray(size);

        Set<String> set = new HashSet<>();
        for (LibFeature feature : features) {
            if (feature.getEnabled() == 1) {
                set.add(feature.getName());
            }
        }
        REFS.recycle(ref);
        return set;
    }

    public static int randomSeed(int seed) {
        return LIB.MXRandomSeed(seed);
    }

    /* Need tests

    public static int randomSeed(int seed, Device device) {
        int deviceType = DeviceType.toDeviceType(device);
        return LIB.MXRandomSeedContext(seed, deviceType, device.getDeviceId());
    }

    public static void notifyShutdown() {
        checkCall(LIB.MXNotifyShutdown());
    }
    */

    /////////////////////////////////
    // Profiler information
    /////////////////////////////////

    /*
    public static int setProcessProfilerConfig(int numParams, String keys[], String vals[],
                                               Pointer kvstoreHandle) {

    }

    int MXSetProfilerConfig(int num_params, String keys[], String vals[]);

    int MXSetProcessProfilerState(int state, int profile_process, Pointer kvStoreHandle);

    int MXSetProfilerState(int state);

    int MXDumpProcessProfile(int finished, int profile_process, Pointer kvStoreHandle);

    int MXDumpProfile(int finished);

    int MXAggregateProfileStatsPrint(String out_str[], int reset);

    int MXProcessProfilePause(int paused, int profile_process, Pointer kvStoreHandle);

    int MXProfilePause(int paused);

    int MXProfileCreateDomain(String domain, PointerByReference out);

    int MXProfileCreateTask(Pointer domain, Pointer task_name, PointerByReference out);

    int MXProfileCreateTask(Pointer domain, String task_name, PointerByReference out);

    int MXProfileCreateFrame(Pointer domain, String frame_name, PointerByReference out);

    int MXProfileCreateEvent(String event_name, PointerByReference out);

    int MXProfileCreateCounter(Pointer domain, String counter_name, PointerByReference out);

    int MXProfileDestroyHandle(Pointer frame_handle);

    int MXProfileDurationStart(Pointer duration_handle);

    int MXProfileDurationStop(Pointer duration_handle);

    int MXProfileSetCounter(Pointer counter_handle, long value);

    int MXProfileAdjustCounter(Pointer counter_handle, long value);

    int MXProfileSetMarker(Pointer domain, String instant_marker_name, String scope);
    */

    /////////////////////////////////
    // NDArray
    /////////////////////////////////

    /* Need tests
    public static Pointer createNdArray() {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArrayCreateNone(ref));

        return ref.getValue();
    }
     */

    public static Pointer createNdArray(
            Device device, Shape shape, DataType dtype, int size, boolean delayedAlloc) {
        int deviceType = MxDeviceType.toDeviceType(device);
        int deviceId = device.getDeviceId();
        int delay = delayedAlloc ? 1 : 0;

        PointerByReference ref = REFS.acquire();
        long[] shapeArray = shape.getShape();
        checkCall(
                LIB.MXNDArrayCreateEx64(
                        shapeArray, size, deviceType, deviceId, delay, dtype.ordinal(), ref));

        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer createSparseNdArray(
            SparseFormat fmt,
            Device device,
            Shape shape,
            DataType dtype,
            DataType[] auxDTypes,
            Shape[] auxShapes,
            boolean delayedAlloc) {
        long[] shapeArray = shape.getShape();
        int deviceType = MxDeviceType.toDeviceType(device);
        int deviceId = device.getDeviceId();
        int delay = delayedAlloc ? 1 : 0;
        PointerByReference ref = REFS.acquire();
        IntBuffer auxDTypesInt =
                IntBuffer.wrap(Arrays.stream(auxDTypes).mapToInt(DataType::ordinal).toArray());
        IntBuffer auxNDims =
                IntBuffer.wrap(Arrays.stream(auxShapes).mapToInt(Shape::dimension).toArray());
        long[] auxShapesInt = Arrays.stream(auxShapes).mapToLong(Shape::head).toArray();
        checkCall(
                LIB.MXNDArrayCreateSparseEx64(
                        fmt.getValue(),
                        shapeArray,
                        shapeArray.length,
                        deviceType,
                        deviceId,
                        delay,
                        dtype.ordinal(),
                        auxDTypes.length,
                        auxDTypesInt,
                        auxNDims,
                        auxShapesInt,
                        ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static void ndArraySyncCopyFromNdArray(MxNDArray dest, MxNDArray src, int location) {
        checkCall(LIB.MXNDArraySyncCopyFromNDArray(dest.getHandle(), src.getHandle(), location));
    }

    /* Need tests
    public static Pointer loadFromBytes(byte[] buf, int offset, int size) {
        Memory memory = new Memory(size);
        memory.write(0, buf, offset, size);

        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArrayLoadFromRawBytes(memory, new NativeSize(size), ref));

        return ref.getValue();
    }

    public static void saveNdArray(String file, Pointer[] ndArrays, String[] keys) {
        PointerArray array = new PointerArray(ndArrays);
        checkCall(LIB.MXNDArraySave(file, ndArrays.length, array, keys));
    }
     */

    public static NDList loadNdArray(MxNDManager manager, Path path, Device device) {
        IntBuffer handlesSize = IntBuffer.allocate(1);
        PointerByReference handlesRef = REFS.acquire();
        PointerByReference namesRef = REFS.acquire();
        IntBuffer namesSize = IntBuffer.allocate(1);
        checkCall(LIB.MXNDArrayLoad(path.toString(), handlesSize, handlesRef, namesSize, namesRef));
        int ndArrayCount = handlesSize.get();
        int nameCount = namesSize.get();
        if (nameCount > 0 && ndArrayCount != nameCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + path.toString());
        }
        Pointer[] handles = handlesRef.getValue().getPointerArray(0, ndArrayCount);
        NDList ndList = new NDList();
        if (nameCount == 0) {
            for (Pointer handle : handles) {
                ndList.add(manager.create(handle));
            }
        } else {
            String[] names = namesRef.getValue().getStringArray(0, nameCount);
            for (int i = 0; i < ndArrayCount; i++) {
                NDArray array = manager.create(handles[i]);
                array.setName(names[i]);
                ndList.add(array);
            }
        }

        REFS.recycle(namesRef);
        REFS.recycle(handlesRef);

        // MXNet always load NDArray on CPU
        if (Device.cpu().equals(device)) {
            return ndList;
        }

        NDList ret = ndList.toDevice(device, true);
        ndList.close();
        return ret;
    }

    /* Need tests
    public static ByteBuffer readBytes(Pointer ndArray) {
        NativeSizeByReference size = new NativeSizeByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArraySaveRawBytes(ndArray, size, ref));

        return ref.getValue().getByteBuffer(0, size.getValue().longValue());
    }
     */

    public static void freeNdArray(Pointer ndArray) {
        checkNDArray(ndArray, "free");
        checkCall(LIB.MXNDArrayFree(ndArray));
    }

    public static void waitToRead(Pointer ndArray) {
        checkNDArray(ndArray, "wait to read");
        checkCall(LIB.MXNDArrayWaitToRead(ndArray));
    }

    public static void waitToWrite(Pointer ndArray) {
        checkNDArray(ndArray, "wait to write");
        checkCall(LIB.MXNDArrayWaitToWrite(ndArray));
    }

    public static void waitAll() {
        checkCall(LIB.MXNDArrayWaitAll());
    }

    public static void syncCopyToCPU(Pointer ndArray, Pointer data, int len) {
        NativeSize size = new NativeSize(len);
        checkNDArray(ndArray, "copy from");
        checkNDArray(data, "copy to");
        checkCall(LIB.MXNDArraySyncCopyToCPU(ndArray, data, size));
    }

    public static void syncCopyFromCPU(Pointer ndArray, Buffer data, int len) {
        NativeSize size = new NativeSize(len);
        Pointer pointer = Native.getDirectBufferPointer(data);
        checkCall(LIB.MXNDArraySyncCopyFromCPU(ndArray, pointer, size));
    }

    public static PairList<Pointer, SparseFormat> imperativeInvoke(
            Pointer function, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        String[] keys;
        String[] values;
        if (params == null) {
            keys = EMPTY_ARRAY;
            values = EMPTY_ARRAY;
        } else {
            keys = params.keyArray(EMPTY_ARRAY);
            values = params.values().stream().map(Object::toString).toArray(String[]::new);
        }
        StringArray keyArray = StringArray.of(keys);
        StringArray valueArray = StringArray.of(values);
        PointerArray srcArray = toPointerArray(src);
        PointerArray destArray = toPointerArray(dest);
        PointerByReference destRef = REFS.acquire();
        destRef.setValue(destArray);
        PointerByReference destSType = REFS.acquire();
        IntBuffer numOutputs = IntBuffer.allocate(1);
        numOutputs.put(0, 1);

        checkCall(
                LIB.MXImperativeInvokeEx(
                        function,
                        src.length,
                        srcArray,
                        numOutputs,
                        destRef,
                        keys.length,
                        keyArray,
                        valueArray,
                        destSType));
        int numOfOutputs = numOutputs.get(0);
        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOfOutputs);
        int[] sTypes = destSType.getValue().getIntArray(0, numOfOutputs);
        PairList<Pointer, SparseFormat> pairList = new PairList<>();
        for (int i = 0; i < numOfOutputs; i++) {
            pairList.add(ptrArray[i], SparseFormat.fromValue(sTypes[i]));
        }
        REFS.recycle(destRef);
        REFS.recycle(destSType);
        srcArray.recycle();
        keyArray.recycle();
        valueArray.recycle();

        if (destArray != null) {
            destArray.recycle();
        }
        return pairList;
    }

    public static SparseFormat getStorageType(Pointer ndArray) {
        IntBuffer type = IntBuffer.allocate(1);
        checkNDArray(ndArray, "get the storage type of");
        checkCall(LIB.MXNDArrayGetStorageType(ndArray, type));
        return SparseFormat.fromValue(type.get());
    }

    public static Device getDevice(Pointer ndArray) {
        IntBuffer deviceType = IntBuffer.allocate(1);
        IntBuffer deviceId = IntBuffer.allocate(1);
        checkNDArray(ndArray, "get the device of");
        checkCall(LIB.MXNDArrayGetContext(ndArray, deviceType, deviceId));
        String deviceTypeStr = MxDeviceType.fromDeviceType(deviceType.get(0));
        // CPU is special case which don't have device id
        return Device.of(deviceTypeStr, deviceId.get(0));
    }

    public static Shape getShape(Pointer ndArray) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();
        checkNDArray(ndArray, "get the shape of");
        checkCall(LIB.MXNDArrayGetShapeEx64(ndArray, dim, ref));
        int nDim = dim.get();
        if (nDim == 0) {
            REFS.recycle(ref);
            return new Shape();
        }
        long[] shape = ref.getValue().getLongArray(0, nDim);
        REFS.recycle(ref);
        return new Shape(shape);
    }

    public static DataType getDataType(Pointer ndArray) {
        IntBuffer dataType = IntBuffer.allocate(1);
        checkNDArray(ndArray, "get the data type of");
        checkCall(LIB.MXNDArrayGetDType(ndArray, dataType));
        return DataType.values()[dataType.get()];
    }

    /* Need tests
    public static DataType getAuxType(Pointer ndArray, int index) {
        IntBuffer dataType = IntBuffer.allocate(1);
        checkCall(LIB.MXNDArrayGetAuxType(ndArray, index, dataType));
        return DataType.values()[dataType.get()];
    }

    public static Pointer getAuxNdArray(Pointer ndArray, int index) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArrayGetAuxNDArray(ndArray, index, ref));
        return ref.getValue();
    }

    public static Pointer getDataNdArray(Pointer ndArray) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArrayGetDataNDArray(ndArray, ref));
        return ref.getValue();
    }

    public static Pointer getGrad(Pointer ndArray) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArrayGetGrad(ndArray, ref));
        return ref.getValue();
    }

    public static Pointer reshape(Pointer ndArray, long[] dims, boolean reverse) {
        PointerByReference ref = new PointerByReference();
        byte reverseByte = reverse ? (byte) 1 : 0;
        checkCall(
                LIB.MXNDArrayReshape64(
                        ndArray, dims.length, LongBuffer.wrap(dims), reverseByte, ref));
        return ref.getValue();
    } */

    /////////////////////////////////
    // MxGradientCollector
    /////////////////////////////////
    public static boolean autogradSetIsRecording(boolean isRecording) {
        IntBuffer prev = IntBuffer.allocate(1);
        checkCall(LIB.MXAutogradSetIsRecording(isRecording ? 1 : 0, prev));
        return prev.get(0) == 1;
    }

    public static boolean autogradSetTraining(boolean isTraining) {
        IntBuffer prev = IntBuffer.allocate(1);
        checkCall(LIB.MXAutogradSetIsTraining(isTraining ? 1 : 0, prev));
        return prev.get(0) == 1;
    }

    public static boolean autogradIsRecording() {
        ByteBuffer isRecording = ByteBuffer.allocate(1);
        checkCall(LIB.MXAutogradIsRecording(isRecording));
        return isRecording.get(0) == 1;
    }

    public static boolean autogradIsTraining() {
        ByteBuffer isTraining = ByteBuffer.allocate(1);
        checkCall(LIB.MXAutogradIsTraining(isTraining));
        return isTraining.get(0) == 1;
    }

    public static void autogradMarkVariables(
            int numVar, Pointer varHandles, IntBuffer reqsArray, Pointer gradHandles) {
        PointerByReference varRef = REFS.acquire();
        PointerByReference gradRef = REFS.acquire();
        varRef.setValue(varHandles);
        gradRef.setValue(gradHandles);
        checkCall(LIB.MXAutogradMarkVariables(numVar, varRef, reqsArray, gradRef));
        REFS.recycle(varRef);
        REFS.recycle(gradRef);
    }

    public static void autogradBackward(NDList array, int retainGraph) {
        PointerByReference ref = REFS.acquire();
        PointerArray pa = toPointerArray(array);
        checkCall(LIB.MXAutogradBackward(array.size(), pa, ref, retainGraph));
        REFS.recycle(ref);
        pa.recycle();
    }

    public static void autogradBackwardExecute(
            int numOutput,
            NDList array,
            NDArray outgrad,
            int numVariables,
            Pointer varHandles,
            int retainGraph,
            int createGraph,
            int isTrain,
            Pointer gradHandles,
            Pointer gradSparseFormat) {
        PointerByReference varRef = REFS.acquire();
        PointerByReference gradRef = REFS.acquire();
        PointerByReference gradSparseFormatRef = REFS.acquire();
        varRef.setValue(varHandles);
        gradRef.setValue(gradHandles);
        gradSparseFormatRef.setValue(gradSparseFormat);
        PointerArray inputHandles = toPointerArray(array);
        PointerArray ogradHandles = PointerArray.of();

        checkCall(
                LIB.MXAutogradBackwardEx(
                        numOutput,
                        inputHandles,
                        ogradHandles,
                        numVariables,
                        varRef,
                        retainGraph,
                        createGraph,
                        isTrain,
                        gradRef,
                        gradSparseFormatRef));
        REFS.recycle(varRef);
        REFS.recycle(gradRef);
        REFS.recycle(gradSparseFormatRef);
        inputHandles.recycle();
        ogradHandles.recycle();
    }

    public static Pointer autogradGetSymbol(NDArray array) {
        Pointer handle = ((MxNDArray) array).getHandle();
        PointerByReference out = REFS.acquire();
        checkCall(LIB.MXAutogradGetSymbol(handle, out));
        Pointer pointer = out.getValue();
        REFS.recycle(out);
        return pointer;
    }

    public static int isNumpyMode() {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(LIB.MXIsNumpyShape(ret));
        return ret.get();
    }

    public static void setNumpyMode(NumpyMode mode) {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(LIB.MXSetIsNumpyShape(mode.ordinal(), ret));
    }

    public static Pointer getGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the gradient for");
        checkCall(LIB.MXNDArrayGetGrad(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer parameterStoreCreate(String type) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXKVStoreCreate(type, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static void parameterStoreClose(Pointer handle) {
        checkCall(LIB.MXKVStoreFree(handle));
    }

    public static void parameterStoreInit(Pointer handle, int num, String[] keys, NDList vals) {
        checkNDArray(handle, "initialize the parameter store with");
        PointerArray pa = toPointerArray(vals);
        checkCall(LIB.MXKVStoreInitEx(handle, num, keys, pa));
        pa.recycle();
    }

    public static void parameterStorePush(
            Pointer handle, int num, String[] keys, NDList vals, int priority) {
        checkNDArray(handle, "push to the parameter store with");
        PointerArray pa = toPointerArray(vals);
        checkCall(LIB.MXKVStorePushEx(handle, num, keys, pa, priority));
        pa.recycle();
    }

    public static void parameterStorePull(
            Pointer handle, int num, int[] keys, NDList vals, int priority) {
        checkNDArray(handle, "pull from the parameter store with");
        PointerArray pa = toPointerArray(vals);
        checkCall(LIB.MXKVStorePull(handle, num, keys, pa, priority));
        pa.recycle();
    }

    public static void parameterStorePull(
            Pointer handle, int num, String[] keys, NDList vals, int priority) {
        checkNDArray(handle, "pull from the parameter store with");
        PointerArray pa = toPointerArray(vals);
        checkCall(LIB.MXKVStorePullEx(handle, num, keys, pa, priority));
        pa.recycle();
    }

    public static void parameterStorePushPull(
            Pointer handle,
            int inputNum,
            String[] inputKeys,
            int outputNum,
            String[] outputKey,
            NDList inputs,
            NDList outputs,
            int priority) {
        checkNDArray(handle, "push from the parameter store with");
        PointerArray inputHandles = toPointerArray(inputs);
        PointerArray outputHandles = toPointerArray(outputs);

        checkCall(
                LIB.MXKVStorePushPullEx(
                        handle,
                        inputNum,
                        inputKeys,
                        outputNum,
                        outputKey,
                        inputHandles,
                        outputHandles,
                        priority));
        inputHandles.recycle();
        outputHandles.recycle();
    }

    public static void parameterStoreSetUpdater(
            Pointer handle,
            MxnetLibrary.MXKVStoreUpdater updater,
            MxnetLibrary.MXKVStoreStrUpdater stringUpdater,
            Pointer updaterHandle) {
        checkCall(LIB.MXKVStoreSetUpdaterEx(handle, updater, stringUpdater, updaterHandle));
    }

    public static void parameterStoreSetUpdater(
            Pointer handle, MxnetLibrary.MXKVStoreUpdater updater, Pointer updaterHandle) {
        checkCall(LIB.MXKVStoreSetUpdater(handle, updater, updaterHandle));
    }

    /*
    int MXInitPSEnv(int num_vars, String keys[], String vals[]);

    int MXKVStoreSetGradientCompression(Pointer handle, int num_params, String keys[],
                                        String vals[]);

    int MXKVStorePullWithSparse(Pointer handle, int num, int keys[], PointerByReference vals,
                                int priority, byte ignore_sparse);

    int MXKVStorePullWithSparseEx(Pointer handle, int num, String keys[], PointerByReference vals,
                                  int priority, byte ignore_sparse);


    int MXKVStorePullRowSparse(Pointer handle, int num, int keys[], PointerByReference vals,
                               PointerByReference row_ids, int priority);

    int MXKVStorePullRowSparseEx(Pointer handle, int num, String keys[], PointerByReference vals,
                                 PointerByReference row_ids, int priority);

    int MXKVStoreGetType(Pointer handle, String type[]);


    int MXKVStoreGetRank(Pointer handle, IntBuffer ret);

    int MXKVStoreGetGroupSize(Pointer handle, IntBuffer ret);

    int MXKVStoreIsWorkerNode(IntBuffer ret);

    int MXKVStoreIsServerNode(IntBuffer ret);

    int MXKVStoreIsSchedulerNode(IntBuffer ret);


    int MXKVStoreBarrier(Pointer handle);


    int MXKVStoreSetBarrierBeforeExit(Pointer handle, int barrier_before_exit);


    int MXKVStoreRunServer(Pointer handle, MxnetLibrary.MXKVStoreServerController controller,
                           Pointer controller_handle);


    int MXKVStoreSendCommmandToServers(Pointer handle, int cmd_id, String cmd_body);

    int MXKVStoreGetNumDeadNode(Pointer handle, int node_id, IntBuffer number, int timeout_sec);
     */
    /*
    int MXImperativeInvokeEx(Pointer creator, int num_inputs, PointerByReference inputs,
                             IntBuffer num_outputs, PointerByReference outputs, int num_params,
                             String param_keys[], String param_vals[],
                             PointerByReference out_stypes);

    int MXNDArraySyncCopyFromCPU(Pointer handle, Pointer data, NativeSize size);

    int MXNDArraySyncCopyFromNDArray(Pointer handle_dst, Pointer handle_src, int i);

    int MXNDArraySyncCheckFormat(Pointer handle, byte full_check);


    int MXNDArrayReshape(Pointer handle, int ndim, IntBuffer dims, PointerByReference out);

    int MXNDArrayReshape64(Pointer handle, int ndim, LongBuffer dims, byte reverse,
                           PointerByReference out);

    int MXNDArrayGetData(Pointer handle, PointerByReference out_pdata);

    int MXNDArrayToDLPack(Pointer handle, PointerByReference out_dlpack);

    int MXNDArrayFromDLPack(Pointer dlpack, PointerByReference out_handle);

    int MXNDArrayCallDLPackDeleter(Pointer dlpack);

    int MXNDArrayGetDType(Pointer handle, IntBuffer out_dtype);

    int MXNDArrayGetAuxType(Pointer handle, int i, IntBuffer out_type);

    int MXNDArrayGetAuxNDArray(Pointer handle, int i, PointerByReference out);

    int MXNDArrayGetDataNDArray(Pointer handle, PointerByReference out);

    int MXNDArrayGetContext(Pointer handle, IntBuffer out_dev_type, IntBuffer out_dev_id);
    */
    public static Pointer detachGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXNDArrayDetach(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }
    /*
    int MXNDArraySetGradState(Pointer handle, int state);

    int MXNDArrayGetGradState(Pointer handle, IntBuffer out);

    int MXListFunctions(IntBuffer out_size, PointerByReference out_array);


    int MXAutogradComputeGradient(int num_output, PointerByReference output_handles);


    int MXAutogradGetSymbol(Pointer handle, PointerByReference out);


    int MXCreateCachedOp(Pointer handle, PointerByReference out);


    int MXCreateCachedOpEx(Pointer handle, int num_flags, String keys[], String vals[],
                           PointerByReference out);


    int MXFreeCachedOp(Pointer handle);


    int MXInvokeCachedOp(Pointer handle, int num_inputs, PointerByReference inputs,
                         IntBuffer num_outputs, PointerByReference outputs);

    int MXInvokeCachedOpEx(Pointer handle, int num_inputs, PointerByReference inputs,
                           IntBuffer num_outputs, PointerByReference outputs,
                           PointerByReference out_stypes);


    int MXListAllOpNames(IntBuffer out_size, PointerByReference out_array);
    */

    /////////////////////////////////
    // MXNet Symbols
    /////////////////////////////////

    public static Pointer getSymbolOutput(Pointer symbol, int index) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolGetOutput(symbol, index, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static String[] listSymbolOutputs(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListOutputs(symbol, size, ref));
        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    /* Need tests
    public static String symbolToJson(Pointer symbol) {
        String[] out = new String[1];
        checkCall(LIB.MXSymbolSaveToJSON(symbol, out));
        return out[0];
    }
     */

    public static void freeSymbol(Pointer symbol) {
        checkCall(LIB.MXSymbolFree(symbol));
    }

    /* Need tests
    public static void saveSymbol(Pointer symbol, String path) {
        checkCall(LIB.MXSymbolSaveToFile(symbol, path));
    }

    public static Pointer copySymbol(Pointer symbol) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCopy(symbol, ref));
        return ref.getValue();
    }

    public static String getSymbolDebugString(Pointer symbol) {
        String[] out = new String[1];
        checkCall(LIB.MXSymbolPrint(symbol, out));
        return out[0];
    }

    public static String getSymbolName(Pointer symbol) {
        String[] out = new String[1];
        IntBuffer success = IntBuffer.allocate(1);
        checkCall(LIB.MXSymbolGetName(symbol, out, success));
        if (success.get() == 1) {
            return out[0];
        }
        return null;
    }

    public static String getSymbolAttr(Pointer symbol, String key) {
        String[] out = new String[1];
        IntBuffer success = IntBuffer.allocate(1);
        checkCall(LIB.MXSymbolGetAttr(symbol, key, out, success));
        if (success.get() == 1) {
            return out[0];
        }
        return null;
    }

    public static void setSymbolAttr(Pointer symbol, String key, String value) {
        checkCall(LIB.MXSymbolSetAttr(symbol, key, value));
    }

    public static PairList<String, String> listSymbolAttr(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolListAttr(symbol, size, ref));

        return toPairList(ref, size.get());
    }

    public static PairList<String, String> listSymbolAttrShallow(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolListAttrShallow(symbol, size, ref));

        return toPairList(ref, size.get());
    }
     */

    public static String[] listSymbolNames(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.NNSymbolListInputNames(symbol, 0, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String[] listSymbolArguments(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListArguments(symbol, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String[] listSymbolAuxiliaryStates(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListAuxiliaryStates(symbol, size, ref));

        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static Pointer getSymbolInternals(Pointer symbol) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolGetInternals(symbol, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    /* Need tests
    public static String[] listSymbolArguments(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolListArguments(symbol, size, ref));

        return toStringArray(ref, size.get());
    }

    public static int getSymbolNumOutputs(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        checkCall(LIB.MXSymbolGetNumOutputs(symbol, size));
        return size.get();
    }

    public static Pointer getSymbolInternals(Pointer symbol) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolGetInternals(symbol, ref));
        return ref.getValue();
    }

    public static String getSymbolChildren(Pointer symbol) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolGetChildren(symbol, ref));
        return ref.getValue().getString(0, StandardCharsets.UTF_8.name());
    }

    public static String[] listSymbolAuxiliaryStates(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolListAuxiliaryStates(symbol, size, ref));

        return toStringArray(ref, size.get());
    }

    public static Pointer[] listAtomicSymbolCreators() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolListAtomicSymbolCreators(outSize, ref));

        int size = outSize.get();
        return ref.getValue().getPointerArray(0, size);
    }

    public static String getAtomicSymbolName(Pointer symbol) {
        String[] ret = new String[1];
        checkCall(LIB.MXSymbolGetAtomicSymbolName(symbol, ret));
        return ret[0];
    }

    public static String getInputSymbols(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolGetInputSymbols(symbol, ref, size));
        return ref.getValue().getString(0, StandardCharsets.UTF_8.name());
    }

    public static String cutSubgraph(Pointer symbol) {
        IntBuffer inputSize = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCutSubgraph(symbol, ref, inputSize));
        return ref.getValue().getString(0, StandardCharsets.UTF_8.name());
    }

    public static Pointer createAtomicSymbol(Pointer symbol, String[] keys, String[] values) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCreateAtomicSymbol(symbol, keys.length, keys, values, ref));
        return ref.getValue();
    }

    public static Pointer createVariable(String name) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCreateVariable(name, ref));
        return ref.getValue();
    }

    public static Pointer createGroup(int numOfSymbols, Pointer symbols) {
        PointerByReference symbolsRef = new PointerByReference(symbols);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCreateGroup(numOfSymbols, symbolsRef, ref));
        return ref.getValue();
    }
     */

    public static Pointer createSymbolFromFile(String path) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromFile(path, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer createSymbolFromString(String json) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromJSON(json, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static String getSymbolString(Pointer symbol) {
        String[] holder = new String[1];
        checkCall(LIB.MXSymbolSaveToJSON(symbol, holder));
        return holder[0];
    }

    private static List<Shape> recoverShape(
            NativeSizeByReference size, PointerByReference nDim, PointerByReference data) {
        int shapeLength = (int) size.getValue().longValue();
        if (shapeLength == 0) {
            return new ArrayList<>();
        }
        int[] dims = nDim.getValue().getIntArray(0, shapeLength);
        int flattenedLength = 0;
        for (int dim : dims) {
            flattenedLength += dim;
        }
        long[] flattenedShapes = data.getValue().getPointer(0).getLongArray(0, flattenedLength);
        int idx = 0;
        List<Shape> result = new ArrayList<>();
        for (int dim : dims) {
            long[] shape = new long[dim];
            System.arraycopy(flattenedShapes, idx, shape, 0, dim);
            idx += dim;
            result.add(new Shape(shape));
        }
        return result;
    }

    public static List<List<Shape>> inferShape(Symbol symbol, PairList<String, Shape> args) {
        Pointer handler = symbol.getHandle();
        int numArgs = args.size();
        String[] keys = args.keys().toArray(new String[0]);
        // the following two is also the representation of
        // CSR NDArray
        long[] indPtr = new long[numArgs + 1];
        Shape flattened = new Shape();
        indPtr[0] = 0;
        for (int i = 0; i < args.size(); i++) {
            Shape shape = args.valueAt(i);
            indPtr[i + 1] = shape.dimension();
            flattened = flattened.addAll(shape);
        }
        long[] flattenedShapeArray = flattened.getShape();

        NativeSizeByReference inShapeSize = new NativeSizeByReference();
        PointerByReference inShapeNDim = REFS.acquire();
        PointerByReference inShapeData = REFS.acquire();
        NativeSizeByReference outShapeSize = new NativeSizeByReference();
        PointerByReference outShapeNDim = REFS.acquire();
        PointerByReference outShapeData = REFS.acquire();
        NativeSizeByReference auxShapeSize = new NativeSizeByReference();
        PointerByReference auxShapeNDim = REFS.acquire();
        PointerByReference auxShapeData = REFS.acquire();
        IntBuffer complete = IntBuffer.allocate(1);
        checkCall(
                LIB.MXSymbolInferShapeEx64(
                        handler,
                        numArgs,
                        keys,
                        indPtr,
                        flattenedShapeArray,
                        inShapeSize,
                        inShapeNDim,
                        inShapeData,
                        outShapeSize,
                        outShapeNDim,
                        outShapeData,
                        auxShapeSize,
                        auxShapeNDim,
                        auxShapeData,
                        complete));
        if (complete.get() != 0) {
            return Arrays.asList(
                    recoverShape(inShapeSize, inShapeNDim, inShapeData),
                    recoverShape(outShapeSize, outShapeNDim, outShapeData),
                    recoverShape(auxShapeSize, auxShapeNDim, auxShapeData));
        }
        return null;
    }

    public static void loadLib(String path, boolean verbose) {
        int intVerbose = verbose ? 1 : 0;
        checkCall(LIB.MXLoadLib(path, intVerbose));
    }

    public static Pointer optimizeFor(Symbol current, String backend, Device device) {
        // TODO: Support partition on parameters
        PointerByReference returnedSymbolHandle = REFS.acquire();
        // placeHolders
        PointerByReference[] placeHolders = {
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire(),
            REFS.acquire()
        };
        // there is no need to update parameters
        checkCall(
                LIB.MXOptimizeForBackend(
                        current.getHandle(),
                        backend,
                        MxDeviceType.toDeviceType(device),
                        returnedSymbolHandle,
                        0,
                        placeHolders[0],
                        0,
                        placeHolders[1],
                        0,
                        new String[0],
                        new String[0],
                        IntBuffer.allocate(1),
                        placeHolders[2],
                        placeHolders[3],
                        IntBuffer.allocate(1),
                        placeHolders[4],
                        placeHolders[5]));
        Pointer ptr = returnedSymbolHandle.getValue();
        REFS.recycle(returnedSymbolHandle);
        Arrays.stream(placeHolders).forEach(REFS::recycle);
        return ptr;
    }

    /* Need tests
    public static Pointer createSymbolFromJson(String json) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXSymbolCreateFromJSON(json, ref));
        return ref.getValue();
    }

    public static Pointer compose(Pointer symbol, String name, String[] keys) {
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolCompose(symbol, name, keys.length, keys, ref));
        return ref.getValue();
    }

    public static Pointer grad(Pointer symbol, String name, int numWrt, String[] wrt) {
        PointerByReference ref = new PointerByReference();

        checkCall(LIB.MXSymbolCompose(symbol, name, numWrt, wrt, ref));
        return ref.getValue();
    }

    public static Shape[] inferShape(Pointer symbol, String[] keys) {
        IntBuffer argIndex = IntBuffer.allocate(1);
        IntBuffer argShapeData = IntBuffer.allocate(1);
        IntBuffer inShapeSize = IntBuffer.allocate(1);
        PointerByReference inShapeNDim = new PointerByReference();
        PointerByReference inShapeData = new PointerByReference();
        IntBuffer outShapeSize = IntBuffer.allocate(1);
        PointerByReference outShapeNDim = new PointerByReference();
        PointerByReference outShapeData = new PointerByReference();
        IntBuffer auxShapeSize = IntBuffer.allocate(1);
        PointerByReference auxShapeNDim = new PointerByReference();
        PointerByReference auxShapeData = new PointerByReference();
        IntBuffer complete = IntBuffer.allocate(1);

        checkCall(
                LIB.MXSymbolInferShape(
                        symbol,
                        keys.length,
                        keys,
                        argIndex.array(),
                        argShapeData.array(),
                        inShapeSize,
                        inShapeNDim,
                        inShapeData,
                        outShapeSize,
                        outShapeNDim,
                        outShapeData,
                        auxShapeSize,
                        auxShapeNDim,
                        auxShapeData,
                        complete));
        if (complete.get() == 1) {
            Shape[] ret = new Shape[keys.length];
            // TODO: add implementation
            return ret; // NOPMD
        }
        return null;
    }

    public static Pointer inferType(Pointer symbol, String[] keys) {
        int[] argTypeData = new int[1];
        IntBuffer inTypeSize = IntBuffer.allocate(1);
        PointerByReference inTypeData = new PointerByReference();
        IntBuffer outTypeSize = IntBuffer.allocate(1);
        PointerByReference outTypeData = new PointerByReference();
        IntBuffer auxTypeSize = IntBuffer.allocate(1);
        PointerByReference auxTypeData = new PointerByReference();
        IntBuffer complete = IntBuffer.allocate(1);

        checkCall(
                LIB.MXSymbolInferType(
                        symbol,
                        keys.length,
                        keys,
                        argTypeData,
                        inTypeSize,
                        inTypeData,
                        outTypeSize,
                        outTypeData,
                        auxTypeSize,
                        auxTypeData,
                        complete));
        if (complete.get() == 1) {
            return outTypeData.getValue();
        }
        return null;
    }

    public static Pointer quantizeSymbol(
            Pointer symbol,
            String[] excludedSymbols,
            String[] offlineParams,
            String quantizedDType,
            byte calibQuantize) {
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXQuantizeSymbol(
                        symbol,
                        ref,
                        excludedSymbols.length,
                        excludedSymbols,
                        offlineParams.length,
                        offlineParams,
                        quantizedDType,
                        calibQuantize));
        return ref.getValue();
    }

    public static Pointer setCalibTableToQuantizedSymbol(
            Pointer symbol,
            String[] layerNames,
            FloatBuffer lowQuantiles,
            FloatBuffer highQuantiles) {
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXSetCalibTableToQuantizedSymbol(
                        symbol, layerNames.length, layerNames, lowQuantiles, highQuantiles, ref));
        return ref.getValue();
    }

    public static Pointer genBackendSubgraph(Pointer symbol, String backend) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXGenBackendSubgraph(symbol, backend, ref));
        return ref.getValue();
    }
     */

    /////////////////////////////////
    // MXNet Executors
    /////////////////////////////////

    /* Need tests
    public static void freeExecutor(Pointer executor) {
        checkCall(LIB.MXExecutorFree(executor));
    }

    public static String getExecutorDebugString(Pointer executor) {
        String[] ret = new String[1];
        checkCall(LIB.MXExecutorPrint(executor, ret));
        return ret[0];
    }

    public static void forward(Pointer executor, boolean isTrain) {
        checkCall(LIB.MXExecutorForward(executor, isTrain ? 1 : 0));
    }

    public static Pointer backward(Pointer executor, int length) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXExecutorBackward(executor, length, ref));
        return ref.getValue();
    }

    public static Pointer backwardEx(Pointer executor, int length, boolean isTrain) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXExecutorBackwardEx(executor, length, ref, isTrain ? 1 : 0));
        return ref.getValue();
    }

    public static NDArray[] getExecutorOutputs(MxNDManager manager, Pointer executor) {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXExecutorOutputs(executor, outSize, ref));
        int size = outSize.get();
        Pointer[] pointers = ref.getValue().getPointerArray(0, size);
        NDArray[] ndArrays = new NDArray[size];
        for (int i = 0; i < size; ++i) {
            ndArrays[i] = manager.create(pointers[i]);
        }
        return ndArrays;
    }

    public static Pointer bindExecutorSimple(
            Symbol symbol,
            Device device,
            String[] g2cKeys,
            int[] g2cDeviceTypes,
            int[] g2cDeviceIds,
            String[] argParams,
            String[] argParamGradReqs,
            String[] inputArgNames,
            IntBuffer inputShapeData,
            IntBuffer inputShapeIdx,
            String[] inputDataTypeNames,
            int[] inputDataTypes,
            String[] inputStorageTypeNames,
            int[] inputStorageTypes,
            String[] sharedArgParams,
            IntBuffer sharedBufferLen,
            String[] sharedBufferNames,
            PointerByReference sharedBufferHandles,
            PointerByReference updatedSharedBufferNames,
            PointerByReference updatedSharedBufferHandles,
            IntBuffer numInArgs,
            PointerByReference inArgs,
            PointerByReference argGrads,
            IntBuffer numAuxStates,
            PointerByReference auxStates) {
        int deviceId = device.getDeviceId();
        int deviceType = DeviceType.toDeviceType(device);

        PointerByReference ref = new PointerByReference();

        checkCall(
                LIB.MXExecutorSimpleBind(
                        symbol.getHandle(),
                        deviceType,
                        deviceId,
                        g2cKeys == null ? 0 : g2cKeys.length,
                        g2cKeys,
                        g2cDeviceTypes,
                        g2cDeviceIds,
                        argParams.length,
                        argParams,
                        argParamGradReqs,
                        inputArgNames.length,
                        inputArgNames,
                        inputShapeData.array(),
                        inputShapeIdx.array(),
                        inputDataTypeNames.length,
                        inputDataTypeNames,
                        inputDataTypes,
                        inputStorageTypeNames == null ? 0 : inputStorageTypeNames.length,
                        inputStorageTypeNames,
                        inputStorageTypes,
                        sharedArgParams.length,
                        sharedArgParams,
                        sharedBufferLen,
                        sharedBufferNames,
                        sharedBufferHandles,
                        updatedSharedBufferNames,
                        updatedSharedBufferHandles,
                        numInArgs,
                        inArgs,
                        argGrads,
                        numAuxStates,
                        auxStates,
                        null,
                        ref));
        return ref.getValue();
    }

    public static Pointer bindExecutor(
            Pointer executor, Device device, int len, int auxStatesLen) {
        int deviceId = device.getDeviceId();
        int deviceType = DeviceType.toDeviceType(device);
        PointerByReference inArgs = new PointerByReference();
        PointerByReference argGradStore = new PointerByReference();
        IntBuffer gradReqType = IntBuffer.allocate(1);
        PointerByReference auxStates = new PointerByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXExecutorBind(
                        executor,
                        deviceType,
                        deviceId,
                        len,
                        inArgs,
                        argGradStore,
                        gradReqType,
                        auxStatesLen,
                        auxStates,
                        ref));
        return ref.getValue();
    }

    public static Pointer bindExecutorX(
            Pointer executor,
            Device device,
            int len,
            int auxStatesLen,
            String[] keys,
            int[] deviceTypes,
            int[] deviceIds) {
        int deviceId = device.getDeviceId();
        int deviceType = DeviceType.toDeviceType(device);
        PointerByReference inArgs = new PointerByReference();
        PointerByReference argGradStore = new PointerByReference();
        IntBuffer gradReqType = IntBuffer.allocate(1);
        PointerByReference auxStates = new PointerByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXExecutorBindX(
                        executor,
                        deviceType,
                        deviceId,
                        keys.length,
                        keys,
                        deviceTypes,
                        deviceIds,
                        len,
                        inArgs,
                        argGradStore,
                        gradReqType,
                        auxStatesLen,
                        auxStates,
                        ref));
        return ref.getValue();
    }

    public static Pointer bindExecutorEX(
            Pointer executor,
            Device device,
            int len,
            int auxStatesLen,
            String[] keys,
            int[] deviceTypes,
            int[] deviceIds,
            Pointer sharedExecutor) {
        int deviceId = device.getDeviceId();
        int deviceType = DeviceType.toDeviceType(device);
        PointerByReference inArgs = new PointerByReference();
        PointerByReference argGradStore = new PointerByReference();
        IntBuffer gradReqType = IntBuffer.allocate(1);
        PointerByReference auxStates = new PointerByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXExecutorBindEX(
                        executor,
                        deviceType,
                        deviceId,
                        keys.length,
                        keys,
                        deviceTypes,
                        deviceIds,
                        len,
                        inArgs,
                        argGradStore,
                        gradReqType,
                        auxStatesLen,
                        auxStates,
                        sharedExecutor,
                        ref));
        return ref.getValue();
    }

    public static Pointer reshapeExecutor(
            boolean partialShaping,
            boolean allowUpSizing,
            Device device,
            String[] keys,
            int[] deviceTypes,
            int[] deviceIds,
            String[] providedArgShapeNames,
            IntBuffer providedArgShapeData,
            IntBuffer providedArgShapeIdx,
            IntBuffer numInArgs,
            PointerByReference inArgs,
            PointerByReference argGrads,
            IntBuffer numAuxStates,
            PointerByReference auxStates,
            Pointer sharedExecutor) {
        int deviceId = device.getDeviceId();
        int deviceType = DeviceType.toDeviceType(device);
        PointerByReference ref = new PointerByReference();
        checkCall(
                LIB.MXExecutorReshape(
                        partialShaping ? 1 : 0,
                        allowUpSizing ? 1 : 0,
                        deviceType,
                        deviceId,
                        keys.length,
                        keys,
                        deviceTypes,
                        deviceIds,
                        providedArgShapeNames.length,
                        providedArgShapeNames,
                        providedArgShapeData.array(),
                        providedArgShapeIdx.array(),
                        numInArgs,
                        inArgs,
                        argGrads,
                        numAuxStates,
                        auxStates,
                        sharedExecutor,
                        ref));
        return ref.getValue();
    }

    public static Pointer getOptimizedSymbol(Pointer executor) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXExecutorGetOptimizedSymbol(executor, ref));
        return ref.getValue();
    }

    public static void setMonitorCallback(
            Pointer executor,
            MxnetLibrary.ExecutorMonitorCallback callback,
            Pointer callbackHandle) {
        checkCall(LIB.MXExecutorSetMonitorCallback(executor, callback, callbackHandle));
    }
     */

    /////////////////////////////////
    // MXNet Executors
    /////////////////////////////////

    /*
    public static Pointer[] listDataIters() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXListDataIters(outSize, ref));
        return ref.getValue().getPointerArray(0, outSize.get());
    }

    public static Pointer createIter(Pointer iter, String[] keys, String[] values) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXDataIterCreateIter(iter, keys.length, keys, values, ref));
        return ref.getValue();
    }

    public static String getIterInfo(Pointer iter) {
        String[] name = new String[1];
        String[] description = new String[1];
        IntBuffer numArgs = IntBuffer.allocate(1);
        PointerByReference argNames = new PointerByReference();
        PointerByReference argTypes = new PointerByReference();
        PointerByReference argDesc = new PointerByReference();
        checkCall(
                LIB.MXDataIterGetIterInfo(
                        iter, name, description, numArgs, argNames, argTypes, argDesc));
        return name[0];
    }

    public static void freeDataIter(Pointer iter) {
        checkCall(LIB.MXDataIterFree(iter));
    }

    public static int next(Pointer iter) {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(LIB.MXDataIterNext(iter, ret));
        return ret.get();
    }

    public static void beforeFirst(Pointer iter) {
        checkCall(LIB.MXDataIterBeforeFirst(iter));
    }

    public static Pointer getData(Pointer iter) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXDataIterGetData(iter, ref));
        return ref.getValue();
    }

    public static Pointer getIndex(Pointer iter) {
        LongBuffer outSize = LongBuffer.wrap(new long[1]);
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXDataIterGetIndex(iter, ref, outSize));
        return ref.getValue();
    }

    public static int getPadNum(Pointer iter) {
        IntBuffer outSize = IntBuffer.allocate(1);
        checkCall(LIB.MXDataIterGetPadNum(iter, outSize));
        return outSize.get();
    }

    public static String getDataIterLabel(Pointer iter) {
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXDataIterGetLabel(iter, ref));
        return ref.getValue().getString(0, StandardCharsets.UTF_8.name());
    }
     */

    /*



    int MXRecordIOWriterCreate(String uri, PointerByReference out);


    int MXRecordIOWriterFree(Pointer handle);


    int MXRecordIOWriterWriteRecord(Pointer handle, String buf, NativeSize size);


    int MXRecordIOWriterTell(Pointer handle, NativeSizeByReference pos);


    int MXRecordIOReaderCreate(String uri, PointerByReference out);


    int MXRecordIOReaderFree(Pointer handle);


    int MXRecordIOReaderReadRecord(Pointer handle, String buf[], NativeSizeByReference size);


    int MXRecordIOReaderSeek(Pointer handle, NativeSize pos);


    int MXRecordIOReaderTell(Pointer handle, NativeSizeByReference pos);


    int MXRtcCreate(ByteBuffer name, int num_input, int num_output, PointerByReference input_names,
                    PointerByReference output_names, PointerByReference inputs,
                    PointerByReference outputs, ByteBuffer kernel, PointerByReference out);


    int MXRtcPush(Pointer handle, int num_input, int num_output, PointerByReference inputs,
                  PointerByReference outputs, int gridDimX, int gridDimY, int gridDimZ,
                  int blockDimX, int blockDimY, int blockDimZ);


    int MXRtcFree(Pointer handle);


    int MXCustomOpRegister(String op_type, MxnetLibrary.CustomOpPropCreator creator);


    int MXCustomFunctionRecord(int num_inputs, PointerByReference inputs, int num_outputs,
                               PointerByReference outputs, MXCallbackList callbacks);


    int MXRtcCudaModuleCreate(String source, int num_options, String options[], int num_exports,
                              String exports[], PointerByReference out);


    int MXRtcCudaModuleFree(Pointer handle);


    int MXRtcCudaKernelCreate(Pointer handle, String name, int num_args, IntBuffer is_ndarray,
                              IntBuffer is_const, IntBuffer arg_types, PointerByReference out);


    int MXRtcCudaKernelFree(Pointer handle);


    int MXRtcCudaKernelCall(Pointer handle, int dev_id, PointerByReference args, int grid_dim_x,
                            int grid_dim_y, int grid_dim_z, int block_dim_x, int block_dim_y,
                            int block_dim_z, int shared_mem);


    int MXNDArrayGetSharedMemHandle(Pointer handle, IntBuffer shared_pid, IntBuffer shared_id);


    int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, IntBuffer shape, int ndim,
                                     int dtype, PointerByReference out);
    */

    //////////////////////////////////
    // cached Op
    //////////////////////////////////

    /**
     * Creates cached op flags.
     *
     * <p>data_indices : [0, 2, 4] Used to label input location, param_indices : [1, 3] Used to
     * label param location
     *
     * @param block the {@link MxSymbolBlock} that loaded in the backend
     * @param manager the NDManager used to create NDArray
     * @param training true if CachedOp is created to forward in traning otherwise, false
     * @return a CachedOp for inference
     */
    public static CachedOp createCachedOp(
            MxSymbolBlock block, MxNDManager manager, boolean training) {
        Symbol symbol = block.getSymbol();

        List<Parameter> parameters = block.getAllParameters();

        // record data index in all inputs
        PairList<String, Integer> dataIndices = new PairList<>();
        // record parameter index in all inputs
        List<Integer> paramIndices = new ArrayList<>();
        int index = 0;
        for (Parameter parameter : parameters) {
            // We assume uninitialized parameters are data inputs
            if (parameter.isInitialized()) {
                paramIndices.add(index);
            } else {
                dataIndices.add(parameter.getName(), index);
            }
            ++index;
        }

        // Creating CachedOp
        Pointer symbolHandle = symbol.getHandle();
        PointerByReference ref = REFS.acquire();

        // static_alloc and static_shape are enabled by default
        String[] keys = {"data_indices", "param_indices", "static_alloc", "static_shape"};
        String[] values = {dataIndices.values().toString(), paramIndices.toString(), "1", "1"};

        checkCall(LIB.MXCreateCachedOpEx(symbolHandle, keys.length, keys, values, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);

        return new CachedOp(pointer, manager, parameters, paramIndices, dataIndices);
    }

    public static void freeCachedOp(Pointer handle) {
        checkCall(LIB.MXFreeCachedOp(handle));
    }

    public static MxNDArray[] cachedOpInvoke(
            MxNDManager manager, Pointer cachedOpHandle, MxNDArray[] inputs) {
        IntBuffer buf = IntBuffer.allocate(1);
        PointerArray array = toPointerArray(inputs);
        PointerByReference ref = REFS.acquire();
        PointerByReference outSTypeRef = REFS.acquire();
        checkCall(
                LIB.MXInvokeCachedOpEx(
                        cachedOpHandle, inputs.length, array, buf, ref, outSTypeRef));
        int numOutputs = buf.get();
        Pointer[] ptrArray = ref.getValue().getPointerArray(0, numOutputs);
        int[] sTypes = outSTypeRef.getValue().getIntArray(0, numOutputs);
        MxNDArray[] output = new MxNDArray[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            if (sTypes[i] != 0) {
                output[i] = manager.create(ptrArray[i], SparseFormat.fromValue(sTypes[i]));
            } else {
                output[i] = manager.create(ptrArray[i]);
            }
        }
        REFS.recycle(ref);
        REFS.recycle(outSTypeRef);
        array.recycle();
        return output;
    }

    public static void checkCall(int ret) {
        if (ret != 0) {
            throw new EngineException("MXNet engine call failed: " + getLastError());
        }
    }

    private static PointerArray toPointerArray(NDList vals) {
        Pointer[] valPointers = new Pointer[vals.size()];
        for (int i = 0; i < vals.size(); i++) {
            valPointers[i] = ((MxNDArray) vals.get(i)).getHandle();
        }
        return PointerArray.of(valPointers);
    }

    private static PointerArray toPointerArray(NDArray[] vals) {
        if (vals == null) {
            return null;
        }
        Pointer[] valPointers = new Pointer[vals.length];
        for (int i = 0; i < vals.length; i++) {
            valPointers[i] = ((MxNDArray) vals[i]).getHandle();
        }
        return PointerArray.of(valPointers);
    }

    private static void checkNDArray(Pointer pointer, String msg) {
        if (pointer == null) {
            throw new IllegalArgumentException(
                    "Tried to " + msg + " an MXNet NDArray that was already closed");
        }
    }

    private static String getLastError() {
        return LIB.MXGetLastError();
    }

    private static String[] toStringArray(PointerByReference ref, int size) {
        if (size == 0) {
            return new String[0];
        }

        Pointer[] pointers = ref.getValue().getPointerArray(0, size);

        String[] arr = new String[size];
        for (int i = 0; i < size; ++i) {
            arr[i] = pointers[i].getString(0, StandardCharsets.UTF_8.name());
        }

        return arr;
    }

    /*
    private static PairList<String, String> toPairList(PointerByReference ref, int size) {
        Pointer[] pointers = ref.getValue().getPointerArray(0, size);

        List<String> names = new ArrayList<>(size);
        List<String> values = new ArrayList<>(size);
        for (Pointer pointer : pointers) {
            String[] pair = pointer.getStringArray(0, 2, StandardCharsets.UTF_8.name());
            names.add(pair[0]);
            values.add(pair[1]);
        }

        return new PairList<>(names, values);
    }
     */

    private static String getOpNamePrefix(String name) {
        for (String prefix : OP_NAME_PREFIX) {
            if (name.startsWith(prefix)) {
                return name.substring(prefix.length());
            }
        }
        return name;
    }
}
