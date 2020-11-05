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
package ai.djl.mxnet.test;

import ai.djl.mxnet.jna.MXCallbackList;
import ai.djl.mxnet.jna.MxnetLibrary;
import ai.djl.mxnet.jna.NativeSize;
import ai.djl.mxnet.jna.NativeSizeByReference;
import ai.djl.mxnet.jna.PointerArray;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

// CHECKSTYLE:OFF:ParameterName
public class MockMxnetLibrary implements MxnetLibrary {

    private Map<String, Function<Object[], Integer>> functions = new HashMap<>();

    public void setFunction(String funcName, Function<Object[], Integer> func) {
        functions.put(funcName, func);
    }

    public void resetFunctions() {
        functions = new HashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public String MXGetLastError() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public int MXLoadLib(String path) {
        if (functions.containsKey("MXLoadLib")) {
            return functions.get("MXLoadLib").apply(new Object[] {path});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXLibInfoFeatures(PointerByReference libFeature, NativeSizeByReference size) {
        if (functions.containsKey("MXLibInfoFeatures")) {
            return functions.get("MXLibInfoFeatures").apply(new Object[] {libFeature, size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRandomSeed(int seed) {
        if (functions.containsKey("MXRandomSeed")) {
            return functions.get("MXRandomSeed").apply(new Object[] {seed});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRandomSeedContext(int seed, int dev_type, int dev_id) {
        if (functions.containsKey("MXRandomSeedContext")) {
            return functions
                    .get("MXRandomSeedContext")
                    .apply(new Object[] {seed, dev_type, dev_id});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNotifyShutdown() {
        if (functions.containsKey("MXNotifyShutdown")) {
            return functions.get("MXNotifyShutdown").apply(new Object[] {});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetProcessProfilerConfig(
            int num_params, String[] keys, String[] vals, Pointer kvstoreHandle) {
        if (functions.containsKey("MXSetProcessProfilerConfig")) {
            return functions
                    .get("MXSetProcessProfilerConfig")
                    .apply(new Object[] {num_params, keys, vals, kvstoreHandle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetProfilerConfig(int num_params, String[] keys, String[] vals) {
        if (functions.containsKey("MXSetProfilerConfig")) {
            return functions
                    .get("MXSetProfilerConfig")
                    .apply(new Object[] {num_params, keys, vals});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetProcessProfilerState(int state, int profile_process, Pointer kvStoreHandle) {
        if (functions.containsKey("MXSetProcessProfilerState")) {
            return functions
                    .get("MXSetProcessProfilerState")
                    .apply(new Object[] {state, profile_process, kvStoreHandle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetProfilerState(int state) {
        if (functions.containsKey("MXSetProfilerState")) {
            return functions.get("MXSetProfilerState").apply(new Object[] {state});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDumpProcessProfile(int finished, int profile_process, Pointer kvStoreHandle) {
        if (functions.containsKey("MXDumpProcessProfile")) {
            return functions
                    .get("MXDumpProcessProfile")
                    .apply(new Object[] {finished, profile_process, kvStoreHandle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDumpProfile(int finished) {
        if (functions.containsKey("MXDumpProfile")) {
            return functions.get("MXDumpProfile").apply(new Object[] {finished});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAggregateProfileStatsPrint(String[] out_str, int reset) {
        if (functions.containsKey("MXAggregateProfileStatsPrint")) {
            return functions
                    .get("MXAggregateProfileStatsPrint")
                    .apply(new Object[] {out_str, reset});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAggregateProfileStatsPrintEx(
            String[] out_str, int reset, int format, int sort_by, int ascending) {
        if (functions.containsKey("MXAggregateProfileStatsPrintEx")) {
            return functions
                    .get("MXAggregateProfileStatsPrintEx")
                    .apply(new Object[] {out_str, reset, format, sort_by, ascending});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProcessProfilePause(int paused, int profile_process, Pointer kvStoreHandle) {
        if (functions.containsKey("MXProcessProfilePause")) {
            return functions
                    .get("MXProcessProfilePause")
                    .apply(new Object[] {paused, profile_process, kvStoreHandle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfilePause(int paused) {
        if (functions.containsKey("MXProfilePause")) {
            return functions.get("MXProfilePause").apply(new Object[] {paused});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileCreateDomain(String domain, PointerByReference out) {
        if (functions.containsKey("MXProfileCreateDomain")) {
            return functions.get("MXProfileCreateDomain").apply(new Object[] {domain, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileCreateTask(Pointer domain, String task_name, PointerByReference out) {
        if (functions.containsKey("MXProfileCreateTask")) {
            return functions
                    .get("MXProfileCreateTask")
                    .apply(new Object[] {domain, task_name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileCreateFrame(Pointer domain, String frame_name, PointerByReference out) {
        if (functions.containsKey("MXProfileCreateFrame")) {
            return functions
                    .get("MXProfileCreateFrame")
                    .apply(new Object[] {domain, frame_name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileCreateEvent(String event_name, PointerByReference out) {
        if (functions.containsKey("MXProfileCreateEvent")) {
            return functions.get("MXProfileCreateEvent").apply(new Object[] {event_name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileCreateCounter(Pointer domain, String counter_name, PointerByReference out) {
        if (functions.containsKey("MXProfileCreateCounter")) {
            return functions
                    .get("MXProfileCreateCounter")
                    .apply(new Object[] {domain, counter_name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileDestroyHandle(Pointer frame_handle) {
        if (functions.containsKey("MXProfileDestroyHandle")) {
            return functions.get("MXProfileDestroyHandle").apply(new Object[] {frame_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileDurationStart(Pointer duration_handle) {
        if (functions.containsKey("MXProfileDurationStart")) {
            return functions.get("MXProfileDurationStart").apply(new Object[] {duration_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileDurationStop(Pointer duration_handle) {
        if (functions.containsKey("MXProfileDurationStop")) {
            return functions.get("MXProfileDurationStop").apply(new Object[] {duration_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileSetCounter(Pointer counter_handle, long value) {
        if (functions.containsKey("MXProfileSetCounter")) {
            return functions.get("MXProfileSetCounter").apply(new Object[] {counter_handle, value});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileAdjustCounter(Pointer counter_handle, long value) {
        if (functions.containsKey("MXProfileAdjustCounter")) {
            return functions
                    .get("MXProfileAdjustCounter")
                    .apply(new Object[] {counter_handle, value});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXProfileSetMarker(Pointer domain, String instant_marker_name, String scope) {
        if (functions.containsKey("MXProfileSetMarker")) {
            return functions
                    .get("MXProfileSetMarker")
                    .apply(new Object[] {domain, instant_marker_name, scope});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetNumOMPThreads(int thread_num) {
        if (functions.containsKey("MXSetNumOMPThreads")) {
            return functions.get("MXSetNumOMPThreads").apply(new Object[] {thread_num});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXEngineSetBulkSize(int bulk_size, IntBuffer prev_bulk_size) {
        if (functions.containsKey("MXEngineSetBulkSize")) {
            return functions
                    .get("MXEngineSetBulkSize")
                    .apply(new Object[] {bulk_size, prev_bulk_size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGetGPUCount(IntBuffer out) {
        if (functions.containsKey("MXGetGPUCount")) {
            return functions.get("MXGetGPUCount").apply(new Object[] {out});
        }
        out.put(0, 1);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGetGPUMemoryInformation(int dev, IntBuffer free_mem, IntBuffer total_mem) {
        if (functions.containsKey("MXGetGPUMemoryInformation")) {
            return functions
                    .get("MXGetGPUMemoryInformation")
                    .apply(new Object[] {dev, free_mem, total_mem});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGetGPUMemoryInformation64(int dev, LongBuffer free_mem, LongBuffer total_mem) {
        if (functions.containsKey("MXGetGPUMemoryInformation64")) {
            return functions
                    .get("MXGetGPUMemoryInformation64")
                    .apply(new Object[] {dev, free_mem, total_mem});
        }

        free_mem.put(900);
        total_mem.put(1000);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGetVersion(IntBuffer out) {
        if (functions.containsKey("MXGetVersion")) {
            return functions.get("MXGetVersion").apply(new Object[] {out});
        }
        out.put(0, 10500);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXLoadTVMOp(String libpath) {
        if (functions.containsKey("MXLoadTVMOp")) {
            return functions.get("MXLoadTVMOp").apply(new Object[] {libpath});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXLoadTVMConfig(PointerByReference config) {
        if (functions.containsKey("MXLoadTVMConfig")) {
            return functions.get("MXLoadTVMConfig").apply(new Object[] {config});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateNone(PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateNone")) {
            return functions.get("MXNDArrayCreateNone").apply(new Object[] {out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreate(
            int[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreate")) {
            return functions
                    .get("MXNDArrayCreate")
                    .apply(new Object[] {shape, ndim, dev_type, dev_id, delay_alloc, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateEx(
            int[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            int dtype,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateEx")) {
            return functions
                    .get("MXNDArrayCreateEx")
                    .apply(new Object[] {shape, ndim, dev_type, dev_id, delay_alloc, dtype, out});
        }

        out.setValue(new PointerArray());
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateEx64(
            long[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            int dtype,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateEx64")) {
            return functions
                    .get("MXNDArrayCreateEx64")
                    .apply(new Object[] {shape, ndim, dev_type, dev_id, delay_alloc, dtype, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateSparseEx(
            int storage_type,
            int[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            int dtype,
            int num_aux,
            IntBuffer aux_type,
            IntBuffer aux_ndims,
            int[] aux_shape,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateSparseEx")) {
            return functions
                    .get("MXNDArrayCreateSparseEx")
                    .apply(
                            new Object[] {
                                storage_type,
                                shape,
                                ndim,
                                dev_type,
                                dev_id,
                                delay_alloc,
                                dtype,
                                num_aux,
                                aux_type,
                                aux_ndims,
                                aux_shape,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateSparseEx64(
            int storage_type,
            long[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            int dtype,
            int num_aux,
            IntBuffer aux_type,
            IntBuffer aux_ndims,
            long[] aux_shape,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateSparseEx64")) {
            return functions
                    .get("MXNDArrayCreateSparseEx64")
                    .apply(
                            new Object[] {
                                storage_type,
                                shape,
                                ndim,
                                dev_type,
                                dev_id,
                                delay_alloc,
                                dtype,
                                num_aux,
                                aux_type,
                                aux_ndims,
                                aux_shape,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayLoadFromRawBytes(Pointer buf, NativeSize size, PointerByReference out) {
        if (functions.containsKey("MXNDArrayLoadFromRawBytes")) {
            return functions.get("MXNDArrayLoadFromRawBytes").apply(new Object[] {buf, size, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySaveRawBytes(
            Pointer handle, NativeSizeByReference out_size, PointerByReference out_buf) {
        if (functions.containsKey("MXNDArraySaveRawBytes")) {
            return functions
                    .get("MXNDArraySaveRawBytes")
                    .apply(new Object[] {handle, out_size, out_buf});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySave(String fname, int num_args, PointerArray args, String[] keys) {
        if (functions.containsKey("MXNDArraySave")) {
            return functions.get("MXNDArraySave").apply(new Object[] {fname, num_args, args, keys});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayLoad(
            String fname,
            IntBuffer out_size,
            PointerByReference out_arr,
            IntBuffer out_name_size,
            PointerByReference out_names) {
        if (functions.containsKey("MXNDArrayLoad")) {
            return functions
                    .get("MXNDArrayLoad")
                    .apply(new Object[] {fname, out_size, out_arr, out_name_size, out_names});
        }

        out_size.put(0, 3);
        out_name_size.put(0, 3);

        PointerArray ndarrays =
                new PointerArray(
                        TestHelper.toPointer("A:" + fname),
                        TestHelper.toPointer("B:b"),
                        TestHelper.toPointer("C:c"));
        PointerArray names =
                new PointerArray(
                        TestHelper.toPointer("A:" + fname),
                        TestHelper.toPointer("B:b"),
                        TestHelper.toPointer("C:c"));

        out_arr.setValue(ndarrays);
        out_names.setValue(names);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayLoadFromBuffer(
            Pointer ndarray_buffer,
            NativeSize size,
            IntBuffer out_size,
            PointerByReference out_arr,
            IntBuffer out_name_size,
            PointerByReference out_names) {
        if (functions.containsKey("MXNDArrayLoadFromBuffer")) {
            return functions
                    .get("MXNDArrayLoadFromBuffer")
                    .apply(
                            new Object[] {
                                ndarray_buffer, size, out_size, out_arr, out_name_size, out_names
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySyncCopyFromCPU(Pointer handle, Pointer data, NativeSize size) {
        if (functions.containsKey("MXNDArraySyncCopyFromCPU")) {
            return functions
                    .get("MXNDArraySyncCopyFromCPU")
                    .apply(new Object[] {handle, data, size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySyncCopyToCPU(Pointer handle, Pointer data, NativeSize size) {
        if (functions.containsKey("MXNDArraySyncCopyToCPU")) {
            return functions.get("MXNDArraySyncCopyToCPU").apply(new Object[] {handle, data, size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySyncCopyFromNDArray(Pointer handle_dst, Pointer handle_src, int i) {
        if (functions.containsKey("MXNDArraySyncCopyFromNDArray")) {
            return functions
                    .get("MXNDArraySyncCopyFromNDArray")
                    .apply(new Object[] {handle_dst, handle_src, i});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySyncCheckFormat(Pointer handle, byte full_check) {
        if (functions.containsKey("MXNDArraySyncCheckFormat")) {
            return functions
                    .get("MXNDArraySyncCheckFormat")
                    .apply(new Object[] {handle, full_check});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayWaitToRead(Pointer handle) {
        if (functions.containsKey("MXNDArrayWaitToRead")) {
            return functions.get("MXNDArrayWaitToRead").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayWaitToWrite(Pointer handle) {
        if (functions.containsKey("MXNDArrayWaitToWrite")) {
            return functions.get("MXNDArrayWaitToWrite").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayWaitAll() {
        if (functions.containsKey("MXNDArrayWaitAll")) {
            return functions.get("MXNDArrayWaitAll").apply(new Object[] {});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayFree(Pointer handle) {
        if (functions.containsKey("MXNDArrayFree")) {
            return functions.get("MXNDArrayFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySlice(
            Pointer handle, int slice_begin, int slice_end, PointerByReference out) {
        if (functions.containsKey("MXNDArraySlice")) {
            return functions
                    .get("MXNDArraySlice")
                    .apply(new Object[] {handle, slice_begin, slice_end, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySlice64(
            Pointer handle, long slice_begin, long slice_end, PointerByReference out) {
        if (functions.containsKey("MXNDArraySlice64")) {
            return functions
                    .get("MXNDArraySlice64")
                    .apply(new Object[] {handle, slice_begin, slice_end, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayAt(Pointer handle, int idx, PointerByReference out) {
        if (functions.containsKey("MXNDArrayAt")) {
            return functions.get("MXNDArrayAt").apply(new Object[] {handle, idx, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayAt64(Pointer handle, long idx, PointerByReference out) {
        if (functions.containsKey("MXNDArrayAt64")) {
            return functions.get("MXNDArrayAt64").apply(new Object[] {handle, idx, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetStorageType(Pointer handle, IntBuffer out_storage_type) {
        if (functions.containsKey("MXNDArrayGetStorageType")) {
            return functions
                    .get("MXNDArrayGetStorageType")
                    .apply(new Object[] {handle, out_storage_type});
        }

        out_storage_type.put(0, 2);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayReshape(Pointer handle, int ndim, IntBuffer dims, PointerByReference out) {
        if (functions.containsKey("MXNDArrayReshape")) {
            return functions.get("MXNDArrayReshape").apply(new Object[] {handle, ndim, dims, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayReshape64(
            Pointer handle, int ndim, LongBuffer dims, byte reverse, PointerByReference out) {
        if (functions.containsKey("MXNDArrayReshape64")) {
            return functions
                    .get("MXNDArrayReshape64")
                    .apply(new Object[] {handle, ndim, dims, reverse, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetShape(Pointer handle, IntBuffer out_dim, PointerByReference out_pdata) {
        if (functions.containsKey("MXNDArrayGetShape")) {
            return functions
                    .get("MXNDArrayGetShape")
                    .apply(new Object[] {handle, out_dim, out_pdata});
        }

        out_dim.put(0, 3);
        Pointer ptr = TestHelper.toPointer(new int[] {1, 2, 3});
        out_pdata.setValue(ptr);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetShapeEx(
            Pointer handle, IntBuffer out_dim, PointerByReference out_pdata) {
        if (functions.containsKey("MXNDArrayGetShapeEx")) {
            return functions
                    .get("MXNDArrayGetShapeEx")
                    .apply(new Object[] {handle, out_dim, out_pdata});
        }

        out_dim.put(0, 3);
        Pointer ptr = TestHelper.toPointer(new int[] {1, 2, 3});
        out_pdata.setValue(ptr);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetShapeEx64(
            Pointer handle, IntBuffer out_dim, PointerByReference out_pdata) {
        if (functions.containsKey("MXNDArrayGetShapeEx64")) {
            return functions
                    .get("MXNDArrayGetShapeEx64")
                    .apply(new Object[] {handle, out_dim, out_pdata});
        }

        out_dim.put(0, 3);
        Pointer ptr = TestHelper.toPointer(new int[] {1, 2, 3});
        out_pdata.setValue(ptr);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetData(Pointer handle, PointerByReference out_pdata) {
        if (functions.containsKey("MXNDArrayGetData")) {
            return functions.get("MXNDArrayGetData").apply(new Object[] {handle, out_pdata});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayToDLPack(Pointer handle, PointerByReference out_dlpack) {
        if (functions.containsKey("MXNDArrayToDLPack")) {
            return functions.get("MXNDArrayToDLPack").apply(new Object[] {handle, out_dlpack});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayFromDLPack(Pointer dlpack, PointerByReference out_handle) {
        if (functions.containsKey("MXNDArrayFromDLPack")) {
            return functions.get("MXNDArrayFromDLPack").apply(new Object[] {dlpack, out_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayFromDLPackEx(
            Pointer dlpack, byte transient_handle, PointerByReference out_handle) {
        if (functions.containsKey("MXNDArrayFromDLPackEx")) {
            return functions
                    .get("MXNDArrayFromDLPackEx")
                    .apply(new Object[] {dlpack, transient_handle, out_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCallDLPackDeleter(Pointer dlpack) {
        if (functions.containsKey("MXNDArrayCallDLPackDeleter")) {
            return functions.get("MXNDArrayCallDLPackDeleter").apply(new Object[] {dlpack});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetDType(Pointer handle, IntBuffer out_dtype) {
        if (functions.containsKey("MXNDArrayGetDType")) {
            return functions.get("MXNDArrayGetDType").apply(new Object[] {handle, out_dtype});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetAuxType(Pointer handle, int i, IntBuffer out_type) {
        if (functions.containsKey("MXNDArrayGetAuxType")) {
            return functions.get("MXNDArrayGetAuxType").apply(new Object[] {handle, i, out_type});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetAuxType64(Pointer handle, long i, IntBuffer out_type) {
        if (functions.containsKey("MXNDArrayGetAuxType64")) {
            return functions.get("MXNDArrayGetAuxType64").apply(new Object[] {handle, i, out_type});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetAuxNDArray(Pointer handle, int i, PointerByReference out) {
        if (functions.containsKey("MXNDArrayGetAuxNDArray")) {
            return functions.get("MXNDArrayGetAuxNDArray").apply(new Object[] {handle, i, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetAuxNDArray64(Pointer handle, long i, PointerByReference out) {
        if (functions.containsKey("MXNDArrayGetAuxNDArray64")) {
            return functions.get("MXNDArrayGetAuxNDArray64").apply(new Object[] {handle, i, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetDataNDArray(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXNDArrayGetDataNDArray")) {
            return functions.get("MXNDArrayGetDataNDArray").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetContext(Pointer handle, IntBuffer out_dev_type, IntBuffer out_dev_id) {
        if (functions.containsKey("MXNDArrayGetContext")) {
            return functions
                    .get("MXNDArrayGetContext")
                    .apply(new Object[] {handle, out_dev_type, out_dev_id});
        }

        out_dev_type.put(0, 2);
        out_dev_id.put(1);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetGrad(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXNDArrayGetGrad")) {
            return functions.get("MXNDArrayGetGrad").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayDetach(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXNDArrayDetach")) {
            return functions.get("MXNDArrayDetach").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArraySetGradState(Pointer handle, int state) {
        if (functions.containsKey("MXNDArraySetGradState")) {
            return functions.get("MXNDArraySetGradState").apply(new Object[] {handle, state});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetGradState(Pointer handle, IntBuffer out) {
        if (functions.containsKey("MXNDArrayGetGradState")) {
            return functions.get("MXNDArrayGetGradState").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXListFunctions(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("MXListFunctions")) {
            return functions.get("MXListFunctions").apply(new Object[] {out_size, out_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGetFunction(String name, PointerByReference out) {
        if (functions.containsKey("MXGetFunction")) {
            return functions.get("MXGetFunction").apply(new Object[] {name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXFuncGetInfo(
            Pointer fun,
            String[] name,
            String[] description,
            IntBuffer num_args,
            PointerByReference arg_names,
            PointerByReference arg_type_infos,
            PointerByReference arg_descriptions,
            String[] return_type) {
        if (functions.containsKey("MXFuncGetInfo")) {
            return functions
                    .get("MXFuncGetInfo")
                    .apply(
                            new Object[] {
                                fun,
                                name,
                                description,
                                num_args,
                                arg_names,
                                arg_type_infos,
                                arg_descriptions,
                                return_type
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXFuncDescribe(
            Pointer fun,
            IntBuffer num_use_vars,
            IntBuffer num_scalars,
            IntBuffer num_mutate_vars,
            IntBuffer type_mask) {
        if (functions.containsKey("MXFuncDescribe")) {
            return functions
                    .get("MXFuncDescribe")
                    .apply(
                            new Object[] {
                                fun, num_use_vars, num_scalars, num_mutate_vars, type_mask
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXFuncInvoke(
            Pointer fun,
            PointerByReference use_vars,
            FloatBuffer scalar_args,
            PointerByReference mutate_vars) {
        if (functions.containsKey("MXFuncInvoke")) {
            return functions
                    .get("MXFuncInvoke")
                    .apply(new Object[] {fun, use_vars, scalar_args, mutate_vars});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXFuncInvokeEx(
            Pointer fun,
            PointerByReference use_vars,
            FloatBuffer scalar_args,
            PointerByReference mutate_vars,
            int num_params,
            PointerByReference param_keys,
            PointerByReference param_vals) {
        if (functions.containsKey("MXFuncInvokeEx")) {
            return functions
                    .get("MXFuncInvokeEx")
                    .apply(
                            new Object[] {
                                fun,
                                use_vars,
                                scalar_args,
                                mutate_vars,
                                num_params,
                                param_keys,
                                param_vals
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXImperativeInvoke(
            Pointer creator,
            int num_inputs,
            PointerArray inputs,
            IntBuffer num_outputs,
            PointerByReference outputs,
            int num_params,
            String[] param_keys,
            String[] param_vals) {
        if (functions.containsKey("MXImperativeInvoke")) {
            return functions
                    .get("MXImperativeInvoke")
                    .apply(
                            new Object[] {
                                creator,
                                num_inputs,
                                inputs,
                                num_outputs,
                                outputs,
                                num_params,
                                param_keys,
                                param_vals
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXImperativeInvokeEx(
            Pointer creator,
            int num_inputs,
            PointerArray inputs,
            IntBuffer num_outputs,
            PointerByReference outputs,
            int num_params,
            String[] param_keys,
            String[] param_vals,
            PointerByReference out_stypes) {
        if (functions.containsKey("MXImperativeInvokeEx")) {
            return functions
                    .get("MXImperativeInvokeEx")
                    .apply(
                            new Object[] {
                                creator,
                                num_inputs,
                                inputs,
                                num_outputs,
                                outputs,
                                num_params,
                                param_keys,
                                param_vals,
                                out_stypes
                            });
        }
        num_outputs.put(1);
        outputs.setValue(new PointerArray(TestHelper.toPointer("test")));
        out_stypes.setValue(TestHelper.toPointer(new int[] {1}));
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradSetIsRecording(int is_recording, IntBuffer prev) {
        if (functions.containsKey("MXAutogradSetIsRecording")) {
            return functions
                    .get("MXAutogradSetIsRecording")
                    .apply(new Object[] {is_recording, prev});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradSetIsTraining(int is_training, IntBuffer prev) {
        if (functions.containsKey("MXAutogradSetIsTraining")) {
            return functions.get("MXAutogradSetIsTraining").apply(new Object[] {is_training, prev});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradIsRecording(ByteBuffer curr) {
        if (functions.containsKey("MXAutogradIsRecording")) {
            return functions.get("MXAutogradIsRecording").apply(new Object[] {curr});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradIsTraining(ByteBuffer curr) {
        if (functions.containsKey("MXAutogradIsTraining")) {
            return functions.get("MXAutogradIsTraining").apply(new Object[] {curr});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXIsNumpyShape(IntBuffer curr) {
        if (functions.containsKey("MXIsNumpyShape")) {
            return functions.get("MXIsNumpyShape").apply(new Object[] {curr});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetIsNumpyShape(int is_np_shape, IntBuffer prev) {
        if (functions.containsKey("MXSetIsNumpyShape")) {
            return functions.get("MXSetIsNumpyShape").apply(new Object[] {is_np_shape, prev});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradMarkVariables(
            int num_var,
            PointerByReference var_handles,
            IntBuffer reqs_array,
            PointerByReference grad_handles) {
        if (functions.containsKey("MXAutogradMarkVariables")) {
            return functions
                    .get("MXAutogradMarkVariables")
                    .apply(new Object[] {num_var, var_handles, reqs_array, grad_handles});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradComputeGradient(int num_output, PointerByReference output_handles) {
        if (functions.containsKey("MXAutogradComputeGradient")) {
            return functions
                    .get("MXAutogradComputeGradient")
                    .apply(new Object[] {num_output, output_handles});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradBackward(
            int num_output,
            PointerArray output_handles,
            PointerByReference ograd_handles,
            int retain_graph) {
        if (functions.containsKey("MXAutogradBackward")) {
            return functions
                    .get("MXAutogradBackward")
                    .apply(new Object[] {num_output, output_handles, ograd_handles, retain_graph});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradBackwardEx(
            int num_output,
            PointerArray output_handles,
            PointerArray ograd_handles,
            int num_variables,
            PointerByReference var_handles,
            int retain_graph,
            int create_graph,
            int is_train,
            PointerByReference grad_handles,
            PointerByReference grad_stypes) {
        if (functions.containsKey("MXAutogradBackwardEx")) {
            return functions
                    .get("MXAutogradBackwardEx")
                    .apply(new Object[] {num_output, output_handles, ograd_handles, retain_graph});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXAutogradGetSymbol(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXAutogradGetSymbol")) {
            return functions.get("MXAutogradGetSymbol").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCreateCachedOp(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXCreateCachedOp")) {
            return functions.get("MXCreateCachedOp").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCreateCachedOpEx(
            Pointer handle, int num_flags, String[] keys, String[] vals, PointerByReference out) {
        if (functions.containsKey("MXCreateCachedOpEx")) {
            return functions
                    .get("MXCreateCachedOpEx")
                    .apply(new Object[] {handle, num_flags, keys, vals, out});
        }

        out.setValue(TestHelper.toPointer("This is a cachedOp"));
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCreateCachedOpEX(
            Pointer handle,
            int num_flags,
            String[] keys,
            String[] vals,
            PointerByReference out,
            byte thread_safe) {
        if (functions.containsKey("MXCreateCachedOpEx")) {
            return functions
                    .get("MXCreateCachedOpEx")
                    .apply(new Object[] {handle, num_flags, keys, vals, out, thread_safe});
        }
        out.setValue(TestHelper.toPointer("This is a cachedOp"));
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXFreeCachedOp(Pointer handle) {
        if (functions.containsKey("MXFreeCachedOp")) {
            return functions.get("MXFreeCachedOp").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXInvokeCachedOp(
            Pointer handle,
            int num_inputs,
            Pointer inputs,
            IntBuffer num_outputs,
            PointerByReference outputs) {
        if (functions.containsKey("MXInvokeCachedOp")) {
            return functions
                    .get("MXInvokeCachedOp")
                    .apply(new Object[] {handle, num_inputs, inputs, num_outputs, outputs});
        }

        num_outputs.put(0, 3);
        PointerArray arr =
                new PointerArray(
                        TestHelper.toPointer("a"),
                        TestHelper.toPointer("b"),
                        TestHelper.toPointer("c"));
        outputs.setValue(arr);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXInvokeCachedOpEx(
            Pointer handle,
            int num_inputs,
            Pointer inputs,
            IntBuffer num_outputs,
            PointerByReference outputs,
            PointerByReference out_stypes) {
        if (functions.containsKey("MXInvokeCachedOpEx")) {
            return functions
                    .get("MXInvokeCachedOpEx")
                    .apply(
                            new Object[] {
                                handle, num_inputs, inputs, num_outputs, outputs, out_stypes
                            });
        }

        num_outputs.put(0, 3);
        PointerArray arr =
                new PointerArray(
                        TestHelper.toPointer("a"),
                        TestHelper.toPointer("b"),
                        TestHelper.toPointer("c"));
        outputs.setValue(arr);
        Pointer sTypes = TestHelper.toPointer(new int[] {0, 0, 1});
        out_stypes.setValue(sTypes);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCachedOpRegisterOpHook(
            Pointer handle, CachedOpMonitorCallback callback, byte monitor_all) {
        if (functions.containsKey("MXCachedOpRegisterOpHook")) {
            return functions
                    .get("MXCachedOpRegisterOpHook")
                    .apply(new Object[] {handle, callback, monitor_all});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXListAllOpNames(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("MXListAllOpNames")) {
            return functions.get("MXListAllOpNames").apply(new Object[] {out_size, out_array});
        }

        PointerArray pa =
                new PointerArray(
                        TestHelper.toPointer("softmax"),
                        TestHelper.toPointer("_npi_copyto"),
                        TestHelper.toPointer("_np_zeros_like"));
        out_size.put(0, 3);
        out_array.setValue(pa);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListAtomicSymbolCreators(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("MXSymbolListAtomicSymbolCreators")) {
            return functions
                    .get("MXSymbolListAtomicSymbolCreators")
                    .apply(new Object[] {out_size, out_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetAtomicSymbolName(Pointer creator, String[] name) {
        if (functions.containsKey("MXSymbolGetAtomicSymbolName")) {
            return functions.get("MXSymbolGetAtomicSymbolName").apply(new Object[] {creator, name});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetInputSymbols(
            Pointer sym, PointerByReference inputs, IntBuffer input_size) {
        if (functions.containsKey("MXSymbolGetInputSymbols")) {
            return functions
                    .get("MXSymbolGetInputSymbols")
                    .apply(new Object[] {sym, inputs, input_size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCutSubgraph(Pointer sym, PointerByReference inputs, IntBuffer input_size) {
        if (functions.containsKey("MXSymbolCutSubgraph")) {
            return functions
                    .get("MXSymbolCutSubgraph")
                    .apply(new Object[] {sym, inputs, input_size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetAtomicSymbolInfo(
            Pointer creator,
            String[] name,
            String[] description,
            IntBuffer num_args,
            PointerByReference arg_names,
            PointerByReference arg_type_infos,
            PointerByReference arg_descriptions,
            String[] key_var_num_args,
            String[] return_type) {
        if (functions.containsKey("MXSymbolGetAtomicSymbolInfo")) {
            return functions
                    .get("MXSymbolGetAtomicSymbolInfo")
                    .apply(
                            new Object[] {
                                creator,
                                name,
                                description,
                                num_args,
                                arg_names,
                                arg_type_infos,
                                arg_descriptions,
                                key_var_num_args,
                                return_type
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCreateAtomicSymbol(
            Pointer creator, int num_param, String[] keys, String[] vals, PointerByReference out) {
        if (functions.containsKey("MXSymbolCreateAtomicSymbol")) {
            return functions
                    .get("MXSymbolCreateAtomicSymbol")
                    .apply(new Object[] {creator, num_param, keys, vals, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCreateVariable(String name, PointerByReference out) {
        if (functions.containsKey("MXSymbolCreateVariable")) {
            return functions.get("MXSymbolCreateVariable").apply(new Object[] {name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCreateGroup(
            int num_symbols, PointerByReference symbols, PointerByReference out) {
        if (functions.containsKey("MXSymbolCreateGroup")) {
            return functions
                    .get("MXSymbolCreateGroup")
                    .apply(new Object[] {num_symbols, symbols, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCreateFromFile(String fname, PointerByReference out) {
        if (functions.containsKey("MXSymbolCreateFromFile")) {
            return functions.get("MXSymbolCreateFromFile").apply(new Object[] {fname, out});
        }

        out.setValue(new PointerArray());
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCreateFromJSON(String json, PointerByReference out) {
        if (functions.containsKey("MXSymbolCreateFromJSON")) {
            return functions.get("MXSymbolCreateFromJSON").apply(new Object[] {json, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolRemoveAmpCast(Pointer sym_handle, PointerByReference ret_sym_handle) {
        if (functions.containsKey("MXSymbolRemoveAmpCast")) {
            return functions
                    .get("MXSymbolRemoveAmpCast")
                    .apply(new Object[] {sym_handle, ret_sym_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolSaveToFile(Pointer symbol, String fname) {
        if (functions.containsKey("MXSymbolSaveToFile")) {
            return functions.get("MXSymbolSaveToFile").apply(new Object[] {symbol, fname});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolSaveToJSON(Pointer symbol, String[] out_json) {
        if (functions.containsKey("MXSymbolSaveToJSON")) {
            return functions.get("MXSymbolSaveToJSON").apply(new Object[] {symbol, out_json});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolFree(Pointer symbol) {
        if (functions.containsKey("MXSymbolFree")) {
            return functions.get("MXSymbolFree").apply(new Object[] {symbol});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCopy(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("MXSymbolCopy")) {
            return functions.get("MXSymbolCopy").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolPrint(Pointer symbol, String[] out_str) {
        if (functions.containsKey("MXSymbolPrint")) {
            return functions.get("MXSymbolPrint").apply(new Object[] {symbol, out_str});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetName(Pointer symbol, String[] out, IntBuffer success) {
        if (functions.containsKey("MXSymbolGetName")) {
            return functions.get("MXSymbolGetName").apply(new Object[] {symbol, out, success});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetAttr(Pointer symbol, String key, String[] out, IntBuffer success) {
        if (functions.containsKey("MXSymbolGetAttr")) {
            return functions.get("MXSymbolGetAttr").apply(new Object[] {symbol, key, out, success});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolSetAttr(Pointer symbol, String key, String value) {
        if (functions.containsKey("MXSymbolSetAttr")) {
            return functions.get("MXSymbolSetAttr").apply(new Object[] {symbol, key, value});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListAttr(Pointer symbol, IntBuffer out_size, PointerByReference out) {
        if (functions.containsKey("MXSymbolListAttr")) {
            return functions.get("MXSymbolListAttr").apply(new Object[] {symbol, out_size, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListAttrShallow(Pointer symbol, IntBuffer out_size, PointerByReference out) {
        if (functions.containsKey("MXSymbolListAttrShallow")) {
            return functions
                    .get("MXSymbolListAttrShallow")
                    .apply(new Object[] {symbol, out_size, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListArguments(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        if (functions.containsKey("MXSymbolListArguments")) {
            return functions
                    .get("MXSymbolListArguments")
                    .apply(new Object[] {symbol, out_size, out_str_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListOutputs(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        if (functions.containsKey("MXSymbolListOutputs")) {
            return functions
                    .get("MXSymbolListOutputs")
                    .apply(new Object[] {symbol, out_size, out_str_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetNumOutputs(Pointer symbol, IntBuffer output_count) {
        if (functions.containsKey("MXSymbolGetNumOutputs")) {
            return functions
                    .get("MXSymbolGetNumOutputs")
                    .apply(new Object[] {symbol, output_count});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetInternals(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("MXSymbolGetInternals")) {
            return functions.get("MXSymbolGetInternals").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetChildren(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("MXSymbolGetChildren")) {
            return functions.get("MXSymbolGetChildren").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGetOutput(Pointer symbol, int index, PointerByReference out) {
        if (functions.containsKey("MXSymbolGetOutput")) {
            return functions.get("MXSymbolGetOutput").apply(new Object[] {symbol, index, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolListAuxiliaryStates(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        if (functions.containsKey("MXSymbolListAuxiliaryStates")) {
            return functions
                    .get("MXSymbolListAuxiliaryStates")
                    .apply(new Object[] {symbol, out_size, out_str_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolCompose(
            Pointer sym, String name, int num_args, String[] keys, PointerByReference args) {
        if (functions.containsKey("MXSymbolCompose")) {
            return functions
                    .get("MXSymbolCompose")
                    .apply(new Object[] {sym, name, num_args, keys, args});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolGrad(Pointer sym, int num_wrt, String[] wrt, PointerByReference out) {
        if (functions.containsKey("MXSymbolGrad")) {
            return functions.get("MXSymbolGrad").apply(new Object[] {sym, num_wrt, wrt, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShape(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            IntBuffer in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            IntBuffer out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            IntBuffer aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShape")) {
            return functions
                    .get("MXSymbolInferShape")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShapeEx(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            IntBuffer in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            IntBuffer out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            IntBuffer aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShapeEx")) {
            return functions
                    .get("MXSymbolInferShapeEx")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShapeEx64(
            Pointer sym,
            int num_args,
            String[] keys,
            long[] arg_ind_ptr,
            long[] arg_shape_data,
            NativeSizeByReference in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            NativeSizeByReference out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            NativeSizeByReference aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShapeEx64")) {
            return functions
                    .get("MXSymbolInferShapeEx64")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShapePartial(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            IntBuffer in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            IntBuffer out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            IntBuffer aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShapePartial")) {
            return functions
                    .get("MXSymbolInferShapePartial")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShapePartialEx(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            IntBuffer in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            IntBuffer out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            IntBuffer aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShapePartialEx")) {
            return functions
                    .get("MXSymbolInferShapePartialEx")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferShapePartialEx64(
            Pointer sym,
            int num_args,
            String[] keys,
            long[] arg_ind_ptr,
            long[] arg_shape_data,
            NativeSizeByReference in_shape_size,
            PointerByReference in_shape_ndim,
            PointerByReference in_shape_data,
            NativeSizeByReference out_shape_size,
            PointerByReference out_shape_ndim,
            PointerByReference out_shape_data,
            NativeSizeByReference aux_shape_size,
            PointerByReference aux_shape_ndim,
            PointerByReference aux_shape_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferShapePartialEx64")) {
            return functions
                    .get("MXSymbolInferShapePartialEx64")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_ind_ptr,
                                arg_shape_data,
                                in_shape_size,
                                in_shape_ndim,
                                in_shape_data,
                                out_shape_size,
                                out_shape_ndim,
                                out_shape_data,
                                aux_shape_size,
                                aux_shape_ndim,
                                aux_shape_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferType(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_type_data,
            IntBuffer in_type_size,
            PointerByReference in_type_data,
            IntBuffer out_type_size,
            PointerByReference out_type_data,
            IntBuffer aux_type_size,
            PointerByReference aux_type_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferType")) {
            return functions
                    .get("MXSymbolInferType")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_type_data,
                                in_type_size,
                                in_type_data,
                                out_type_size,
                                out_type_data,
                                aux_type_size,
                                aux_type_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSymbolInferTypePartial(
            Pointer sym,
            int num_args,
            String[] keys,
            int[] arg_type_data,
            IntBuffer in_type_size,
            PointerByReference in_type_data,
            IntBuffer out_type_size,
            PointerByReference out_type_data,
            IntBuffer aux_type_size,
            PointerByReference aux_type_data,
            IntBuffer complete) {
        if (functions.containsKey("MXSymbolInferTypePartial")) {
            return functions
                    .get("MXSymbolInferTypePartial")
                    .apply(
                            new Object[] {
                                sym,
                                num_args,
                                keys,
                                arg_type_data,
                                in_type_size,
                                in_type_data,
                                out_type_size,
                                out_type_data,
                                aux_type_size,
                                aux_type_data,
                                complete
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXQuantizeSymbol(
            Pointer sym_handle,
            PointerByReference ret_sym_handle,
            int[] dev_type,
            int num_excluded_sym_names,
            String[] excluded_sym_names,
            int num_excluded_op_names,
            String[] excluded_op_names,
            int num_offline,
            String[] offline_params,
            String quantized_dtype,
            byte calib_quantize,
            String quantize_mode,
            String quantize_granularity,
            IntBuffer out_num_calib_names,
            PointerByReference out_calib_names) {
        if (functions.containsKey("MXQuantizeSymbol")) {
            return functions
                    .get("MXQuantizeSymbol")
                    .apply(
                            new Object[] {
                                sym_handle,
                                ret_sym_handle,
                                dev_type,
                                num_excluded_sym_names,
                                excluded_sym_names,
                                num_excluded_op_names,
                                excluded_op_names,
                                num_offline,
                                offline_params,
                                quantized_dtype,
                                calib_quantize,
                                quantize_mode,
                                quantize_granularity,
                                out_num_calib_names,
                                out_calib_names
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXReducePrecisionSymbol(
            Pointer sym_handle,
            PointerByReference ret_sym_handle,
            int num_args,
            int[] arg_type_data,
            int num_ind_ptr,
            int[] ind_ptr,
            int[] target_dtype,
            int cast_optional_params,
            int num_target_dtype_op_names,
            int num_fp32_op_names,
            int num_widest_dtype_op_names,
            int num_conditional_fp32_op_names,
            int num_excluded_symbols,
            int num_model_params,
            String[] target_dtype_op_names,
            String[] fp32_op_names,
            String[] widest_dtype_op_names,
            String[] conditional_fp32_op_names,
            String[] excluded_symbols,
            String[] conditional_param_names,
            String[] conditional_param_vals,
            String[] model_param_names,
            String[] arg_names) {
        if (functions.containsKey("MXReducePrecisionSymbol")) {
            return functions
                    .get("MXReducePrecisionSymbol")
                    .apply(
                            new Object[] {
                                sym_handle,
                                ret_sym_handle,
                                num_args,
                                arg_type_data,
                                num_ind_ptr,
                                ind_ptr,
                                target_dtype,
                                cast_optional_params,
                                num_target_dtype_op_names,
                                num_fp32_op_names,
                                num_widest_dtype_op_names,
                                num_conditional_fp32_op_names,
                                num_excluded_symbols,
                                num_model_params,
                                target_dtype_op_names,
                                fp32_op_names,
                                widest_dtype_op_names,
                                conditional_fp32_op_names,
                                excluded_symbols,
                                conditional_param_names,
                                conditional_param_vals,
                                model_param_names,
                                arg_names
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXSetCalibTableToQuantizedSymbol(
            Pointer qsym_handle,
            int num_layers,
            String[] layer_names,
            FloatBuffer low_quantiles,
            FloatBuffer high_quantiles,
            PointerByReference ret_sym_handle) {
        if (functions.containsKey("MXSetCalibTableToQuantizedSymbol")) {
            return functions
                    .get("MXSetCalibTableToQuantizedSymbol")
                    .apply(
                            new Object[] {
                                qsym_handle,
                                num_layers,
                                layer_names,
                                low_quantiles,
                                high_quantiles,
                                ret_sym_handle
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGenBackendSubgraph(
            Pointer sym_handle, String backend, PointerByReference ret_sym_handle) {
        if (functions.containsKey("MXGenBackendSubgraph")) {
            return functions
                    .get("MXGenBackendSubgraph")
                    .apply(new Object[] {sym_handle, backend, ret_sym_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXGenAtomicSymbolFromSymbol(Pointer sym_handle, PointerByReference ret_sym_handle) {
        if (functions.containsKey("MXGenAtomicSymbolFromSymbol")) {
            return functions
                    .get("MXGenAtomicSymbolFromSymbol")
                    .apply(new Object[] {sym_handle, ret_sym_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXOptimizeForBackend(
            Pointer sym_handle,
            String backend_name,
            int dev_type,
            PointerByReference ret_sym_handle,
            int len,
            PointerByReference in_args_handle,
            int num_options,
            String[] keys,
            String[] vals) {
        if (functions.containsKey("MXOptimizeForBackend")) {
            return functions
                    .get("MXOptimizeForBackend")
                    .apply(
                            new Object[] {
                                sym_handle,
                                backend_name,
                                dev_type,
                                ret_sym_handle,
                                len,
                                in_args_handle,
                                num_options,
                                keys,
                                vals
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorFree(Pointer handle) {
        if (functions.containsKey("MXExecutorFree")) {
            return functions.get("MXExecutorFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorPrint(Pointer handle, String[] out_str) {
        if (functions.containsKey("MXExecutorPrint")) {
            return functions.get("MXExecutorPrint").apply(new Object[] {handle, out_str});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorForward(Pointer handle, int is_train) {
        if (functions.containsKey("MXExecutorForward")) {
            return functions.get("MXExecutorForward").apply(new Object[] {handle, is_train});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorBackward(Pointer handle, int len, PointerByReference head_grads) {
        if (functions.containsKey("MXExecutorBackward")) {
            return functions
                    .get("MXExecutorBackward")
                    .apply(new Object[] {handle, len, head_grads});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorBackwardEx(
            Pointer handle, int len, PointerByReference head_grads, int is_train) {
        if (functions.containsKey("MXExecutorBackwardEx")) {
            return functions
                    .get("MXExecutorBackwardEx")
                    .apply(new Object[] {handle, len, head_grads, is_train});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorOutputs(Pointer handle, IntBuffer out_size, PointerByReference out) {
        if (functions.containsKey("MXExecutorOutputs")) {
            return functions.get("MXExecutorOutputs").apply(new Object[] {handle, out_size, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorBind(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int len,
            PointerByReference in_args,
            PointerByReference arg_grad_store,
            IntBuffer grad_req_type,
            int aux_states_len,
            PointerByReference aux_states,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorBind")) {
            return functions
                    .get("MXExecutorBind")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                len,
                                in_args,
                                arg_grad_store,
                                grad_req_type,
                                aux_states_len,
                                aux_states,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorBindX(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int num_map_keys,
            String[] map_keys,
            int[] map_dev_types,
            int[] map_dev_ids,
            int len,
            PointerByReference in_args,
            PointerByReference arg_grad_store,
            IntBuffer grad_req_type,
            int aux_states_len,
            PointerByReference aux_states,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorBindX")) {
            return functions
                    .get("MXExecutorBindX")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                num_map_keys,
                                map_keys,
                                map_dev_types,
                                map_dev_ids,
                                len,
                                in_args,
                                arg_grad_store,
                                grad_req_type,
                                aux_states_len,
                                aux_states,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorBindEX(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int num_map_keys,
            String[] map_keys,
            int[] map_dev_types,
            int[] map_dev_ids,
            int len,
            PointerByReference in_args,
            PointerByReference arg_grad_store,
            IntBuffer grad_req_type,
            int aux_states_len,
            PointerByReference aux_states,
            Pointer shared_exec,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorBindEX")) {
            return functions
                    .get("MXExecutorBindEX")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                num_map_keys,
                                map_keys,
                                map_dev_types,
                                map_dev_ids,
                                len,
                                in_args,
                                arg_grad_store,
                                grad_req_type,
                                aux_states_len,
                                aux_states,
                                shared_exec,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorSimpleBind(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int num_g2c_keys,
            String[] g2c_keys,
            int[] g2c_dev_types,
            int[] g2c_dev_ids,
            int provided_grad_req_list_len,
            String[] provided_grad_req_names,
            String[] provided_grad_req_types,
            int num_provided_arg_shapes,
            String[] provided_arg_shape_names,
            int[] provided_arg_shape_data,
            int[] provided_arg_shape_idx,
            int num_provided_arg_dtypes,
            String[] provided_arg_dtype_names,
            int[] provided_arg_dtypes,
            int num_provided_arg_stypes,
            String[] provided_arg_stype_names,
            int[] provided_arg_stypes,
            int num_shared_arg_names,
            String[] shared_arg_name_list,
            IntBuffer shared_buffer_len,
            String[] shared_buffer_name_list,
            PointerByReference shared_buffer_handle_list,
            PointerByReference updated_shared_buffer_name_list,
            PointerByReference updated_shared_buffer_handle_list,
            IntBuffer num_in_args,
            PointerByReference in_args,
            PointerByReference arg_grads,
            IntBuffer num_aux_states,
            PointerByReference aux_states,
            Pointer shared_exec_handle,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorSimpleBind")) {
            return functions
                    .get("MXExecutorSimpleBind")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                num_g2c_keys,
                                g2c_keys,
                                g2c_dev_types,
                                g2c_dev_ids,
                                provided_grad_req_list_len,
                                provided_grad_req_names,
                                provided_grad_req_types,
                                num_provided_arg_shapes,
                                provided_arg_shape_names,
                                provided_arg_shape_data,
                                provided_arg_shape_idx,
                                num_provided_arg_dtypes,
                                provided_arg_dtype_names,
                                provided_arg_dtypes,
                                num_provided_arg_stypes,
                                provided_arg_stype_names,
                                provided_arg_stypes,
                                num_shared_arg_names,
                                shared_arg_name_list,
                                shared_buffer_len,
                                shared_buffer_name_list,
                                shared_buffer_handle_list,
                                updated_shared_buffer_name_list,
                                updated_shared_buffer_handle_list,
                                num_in_args,
                                in_args,
                                arg_grads,
                                num_aux_states,
                                aux_states,
                                shared_exec_handle,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorSimpleBindEx(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int num_g2c_keys,
            String[] g2c_keys,
            int[] g2c_dev_types,
            int[] g2c_dev_ids,
            int provided_grad_req_list_len,
            String[] provided_grad_req_names,
            String[] provided_grad_req_types,
            int num_provided_arg_shapes,
            String[] provided_arg_shape_names,
            int[] provided_arg_shape_data,
            int[] provided_arg_shape_idx,
            int num_provided_arg_dtypes,
            String[] provided_arg_dtype_names,
            int[] provided_arg_dtypes,
            int num_provided_arg_stypes,
            String[] provided_arg_stype_names,
            int[] provided_arg_stypes,
            int num_shared_arg_names,
            String[] shared_arg_name_list,
            IntBuffer shared_buffer_len,
            String[] shared_buffer_name_list,
            PointerByReference shared_buffer_handle_list,
            PointerByReference updated_shared_buffer_name_list,
            PointerByReference updated_shared_buffer_handle_list,
            IntBuffer num_in_args,
            PointerByReference in_args,
            PointerByReference arg_grads,
            IntBuffer num_aux_states,
            PointerByReference aux_states,
            Pointer shared_exec_handle,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorSimpleBindEx")) {
            return functions
                    .get("MXExecutorSimpleBindEx")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                num_g2c_keys,
                                g2c_keys,
                                g2c_dev_types,
                                g2c_dev_ids,
                                provided_grad_req_list_len,
                                provided_grad_req_names,
                                provided_grad_req_types,
                                num_provided_arg_shapes,
                                provided_arg_shape_names,
                                provided_arg_shape_data,
                                provided_arg_shape_idx,
                                num_provided_arg_dtypes,
                                provided_arg_dtype_names,
                                provided_arg_dtypes,
                                num_provided_arg_stypes,
                                provided_arg_stype_names,
                                provided_arg_stypes,
                                num_shared_arg_names,
                                shared_arg_name_list,
                                shared_buffer_len,
                                shared_buffer_name_list,
                                shared_buffer_handle_list,
                                updated_shared_buffer_name_list,
                                updated_shared_buffer_handle_list,
                                num_in_args,
                                in_args,
                                arg_grads,
                                num_aux_states,
                                aux_states,
                                shared_exec_handle,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorSimpleBindEx64(
            Pointer symbol_handle,
            int dev_type,
            int dev_id,
            int num_g2c_keys,
            String[] g2c_keys,
            int[] g2c_dev_types,
            int[] g2c_dev_ids,
            int provided_grad_req_list_len,
            String[] provided_grad_req_names,
            String[] provided_grad_req_types,
            int num_provided_arg_shapes,
            String[] provided_arg_shape_names,
            long[] provided_arg_shape_data,
            int[] provided_arg_shape_idx,
            int num_provided_arg_dtypes,
            String[] provided_arg_dtype_names,
            int[] provided_arg_dtypes,
            int num_provided_arg_stypes,
            String[] provided_arg_stype_names,
            int[] provided_arg_stypes,
            int num_shared_arg_names,
            String[] shared_arg_name_list,
            IntBuffer shared_buffer_len,
            String[] shared_buffer_name_list,
            PointerByReference shared_buffer_handle_list,
            PointerByReference updated_shared_buffer_name_list,
            PointerByReference updated_shared_buffer_handle_list,
            IntBuffer num_in_args,
            PointerByReference in_args,
            PointerByReference arg_grads,
            IntBuffer num_aux_states,
            PointerByReference aux_states,
            Pointer shared_exec_handle,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorSimpleBindEx64")) {
            return functions
                    .get("MXExecutorSimpleBindEx64")
                    .apply(
                            new Object[] {
                                symbol_handle,
                                dev_type,
                                dev_id,
                                num_g2c_keys,
                                g2c_keys,
                                g2c_dev_types,
                                g2c_dev_ids,
                                provided_grad_req_list_len,
                                provided_grad_req_names,
                                provided_grad_req_types,
                                num_provided_arg_shapes,
                                provided_arg_shape_names,
                                provided_arg_shape_data,
                                provided_arg_shape_idx,
                                num_provided_arg_dtypes,
                                provided_arg_dtype_names,
                                provided_arg_dtypes,
                                num_provided_arg_stypes,
                                provided_arg_stype_names,
                                provided_arg_stypes,
                                num_shared_arg_names,
                                shared_arg_name_list,
                                shared_buffer_len,
                                shared_buffer_name_list,
                                shared_buffer_handle_list,
                                updated_shared_buffer_name_list,
                                updated_shared_buffer_handle_list,
                                num_in_args,
                                in_args,
                                arg_grads,
                                num_aux_states,
                                aux_states,
                                shared_exec_handle,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorReshape(
            int partial_shaping,
            int allow_up_sizing,
            int dev_type,
            int dev_id,
            int num_map_keys,
            String[] map_keys,
            int[] map_dev_types,
            int[] map_dev_ids,
            int num_provided_arg_shapes,
            String[] provided_arg_shape_names,
            int[] provided_arg_shape_data,
            int[] provided_arg_shape_idx,
            IntBuffer num_in_args,
            PointerByReference in_args,
            PointerByReference arg_grads,
            IntBuffer num_aux_states,
            PointerByReference aux_states,
            Pointer shared_exec,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorReshape")) {
            return functions
                    .get("MXExecutorReshape")
                    .apply(
                            new Object[] {
                                partial_shaping,
                                allow_up_sizing,
                                dev_type,
                                dev_id,
                                num_map_keys,
                                map_keys,
                                map_dev_types,
                                map_dev_ids,
                                num_provided_arg_shapes,
                                provided_arg_shape_names,
                                provided_arg_shape_data,
                                provided_arg_shape_idx,
                                num_in_args,
                                in_args,
                                arg_grads,
                                num_aux_states,
                                aux_states,
                                shared_exec,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorReshapeEx(
            int partial_shaping,
            int allow_up_sizing,
            int dev_type,
            int dev_id,
            int num_map_keys,
            String[] map_keys,
            int[] map_dev_types,
            int[] map_dev_ids,
            int num_provided_arg_shapes,
            String[] provided_arg_shape_names,
            int[] provided_arg_shape_data,
            int[] provided_arg_shape_idx,
            IntBuffer num_in_args,
            PointerByReference in_args,
            PointerByReference arg_grads,
            IntBuffer num_aux_states,
            PointerByReference aux_states,
            Pointer shared_exec,
            PointerByReference out) {
        if (functions.containsKey("MXExecutorReshapeEx")) {
            return functions
                    .get("MXExecutorReshapeEx")
                    .apply(
                            new Object[] {
                                partial_shaping,
                                allow_up_sizing,
                                dev_type,
                                dev_id,
                                num_map_keys,
                                map_keys,
                                map_dev_types,
                                map_dev_ids,
                                num_provided_arg_shapes,
                                provided_arg_shape_names,
                                provided_arg_shape_data,
                                provided_arg_shape_idx,
                                num_in_args,
                                in_args,
                                arg_grads,
                                num_aux_states,
                                aux_states,
                                shared_exec,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorGetOptimizedSymbol(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXExecutorGetOptimizedSymbol")) {
            return functions.get("MXExecutorGetOptimizedSymbol").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorSetMonitorCallback(
            Pointer handle, ExecutorMonitorCallback callback, Pointer callback_handle) {
        if (functions.containsKey("MXExecutorSetMonitorCallback")) {
            return functions
                    .get("MXExecutorSetMonitorCallback")
                    .apply(new Object[] {handle, callback, callback_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXExecutorSetMonitorCallbackEX(
            Pointer handle,
            ExecutorMonitorCallback callback,
            Pointer callback_handle,
            byte monitor_all) {
        if (functions.containsKey("MXExecutorSetMonitorCallbackEX")) {
            return functions
                    .get("MXExecutorSetMonitorCallbackEX")
                    .apply(new Object[] {handle, callback, callback_handle, monitor_all});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXListDataIters(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("MXListDataIters")) {
            return functions.get("MXListDataIters").apply(new Object[] {out_size, out_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterCreateIter(
            Pointer handle, int num_param, String[] keys, String[] vals, PointerByReference out) {
        if (functions.containsKey("MXDataIterCreateIter")) {
            return functions
                    .get("MXDataIterCreateIter")
                    .apply(new Object[] {handle, num_param, keys, vals, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterGetIterInfo(
            Pointer creator,
            String[] name,
            String[] description,
            IntBuffer num_args,
            PointerByReference arg_names,
            PointerByReference arg_type_infos,
            PointerByReference arg_descriptions) {
        if (functions.containsKey("MXDataIterGetIterInfo")) {
            return functions
                    .get("MXDataIterGetIterInfo")
                    .apply(
                            new Object[] {
                                creator,
                                name,
                                description,
                                num_args,
                                arg_names,
                                arg_type_infos,
                                arg_descriptions
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterFree(Pointer handle) {
        if (functions.containsKey("MXDataIterFree")) {
            return functions.get("MXDataIterFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterNext(Pointer handle, IntBuffer out) {
        if (functions.containsKey("MXDataIterNext")) {
            return functions.get("MXDataIterNext").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterBeforeFirst(Pointer handle) {
        if (functions.containsKey("MXDataIterBeforeFirst")) {
            return functions.get("MXDataIterBeforeFirst").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterGetData(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXDataIterGetData")) {
            return functions.get("MXDataIterGetData").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterGetIndex(
            Pointer handle, PointerByReference out_index, LongBuffer out_size) {
        if (functions.containsKey("MXDataIterGetIndex")) {
            return functions
                    .get("MXDataIterGetIndex")
                    .apply(new Object[] {handle, out_index, out_size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterGetPadNum(Pointer handle, IntBuffer pad) {
        if (functions.containsKey("MXDataIterGetPadNum")) {
            return functions.get("MXDataIterGetPadNum").apply(new Object[] {handle, pad});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXDataIterGetLabel(Pointer handle, PointerByReference out) {
        if (functions.containsKey("MXDataIterGetLabel")) {
            return functions.get("MXDataIterGetLabel").apply(new Object[] {handle, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXInitPSEnv(int num_vars, String[] keys, String[] vals) {
        if (functions.containsKey("MXInitPSEnv")) {
            return functions.get("MXInitPSEnv").apply(new Object[] {num_vars, keys, vals});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreCreate(String type, PointerByReference out) {
        if (functions.containsKey("MXKVStoreCreate")) {
            return functions.get("MXKVStoreCreate").apply(new Object[] {type, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreSetGradientCompression(
            Pointer handle, int num_params, String[] keys, String[] vals) {
        if (functions.containsKey("MXKVStoreSetGradientCompression")) {
            return functions
                    .get("MXKVStoreSetGradientCompression")
                    .apply(new Object[] {handle, num_params, keys, vals});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreFree(Pointer handle) {
        if (functions.containsKey("MXKVStoreFree")) {
            return functions.get("MXKVStoreFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreInit(Pointer handle, int num, int[] keys, PointerArray vals) {
        if (functions.containsKey("MXKVStoreInit")) {
            return functions.get("MXKVStoreInit").apply(new Object[] {handle, num, keys, vals});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreInitEx(Pointer handle, int num, String[] keys, PointerArray vals) {
        if (functions.containsKey("MXKVStoreInitEx")) {
            return functions.get("MXKVStoreInitEx").apply(new Object[] {handle, num, keys, vals});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePush(Pointer handle, int num, int[] keys, PointerArray vals, int priority) {
        if (functions.containsKey("MXKVStorePush")) {
            return functions
                    .get("MXKVStorePush")
                    .apply(new Object[] {handle, num, keys, vals, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePushEx(
            Pointer handle, int num, String[] keys, PointerArray vals, int priority) {
        if (functions.containsKey("MXKVStorePushEx")) {
            return functions
                    .get("MXKVStorePushEx")
                    .apply(new Object[] {handle, num, keys, vals, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePullWithSparse(
            Pointer handle,
            int num,
            int[] keys,
            PointerByReference vals,
            int priority,
            byte ignore_sparse) {
        if (functions.containsKey("MXKVStorePullWithSparse")) {
            return functions
                    .get("MXKVStorePullWithSparse")
                    .apply(new Object[] {handle, num, keys, vals, priority, ignore_sparse});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePullWithSparseEx(
            Pointer handle,
            int num,
            String[] keys,
            PointerByReference vals,
            int priority,
            byte ignore_sparse) {
        if (functions.containsKey("MXKVStorePullWithSparseEx")) {
            return functions
                    .get("MXKVStorePullWithSparseEx")
                    .apply(new Object[] {handle, num, keys, vals, priority, ignore_sparse});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePull(Pointer handle, int num, int[] keys, PointerArray vals, int priority) {
        if (functions.containsKey("MXKVStorePull")) {
            return functions
                    .get("MXKVStorePull")
                    .apply(new Object[] {handle, num, keys, vals, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePullEx(
            Pointer handle, int num, String[] keys, PointerArray vals, int priority) {
        if (functions.containsKey("MXKVStorePullEx")) {
            return functions
                    .get("MXKVStorePullEx")
                    .apply(new Object[] {handle, num, keys, vals, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePullRowSparse(
            Pointer handle,
            int num,
            int[] keys,
            PointerByReference vals,
            PointerByReference row_ids,
            int priority) {
        if (functions.containsKey("MXKVStorePullRowSparse")) {
            return functions
                    .get("MXKVStorePullRowSparse")
                    .apply(new Object[] {handle, num, keys, vals, row_ids, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePullRowSparseEx(
            Pointer handle,
            int num,
            String[] keys,
            PointerByReference vals,
            PointerByReference row_ids,
            int priority) {
        if (functions.containsKey("MXKVStorePullRowSparseEx")) {
            return functions
                    .get("MXKVStorePullRowSparseEx")
                    .apply(new Object[] {handle, num, keys, vals, row_ids, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePushPull(
            Pointer handle,
            int vnum,
            int[] vkeys,
            int onum,
            int[] okeys,
            PointerByReference vals,
            PointerByReference outs,
            int priority) {
        if (functions.containsKey("MXKVStorePushPull")) {
            return functions
                    .get("MXKVStorePushPull")
                    .apply(new Object[] {handle, vnum, vkeys, onum, okeys, vals, outs, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStorePushPullEx(
            Pointer handle,
            int vnum,
            String[] vkeys,
            int onum,
            String[] okeys,
            PointerArray vals,
            PointerArray outs,
            int priority) {
        if (functions.containsKey("MXKVStorePushPullEx")) {
            return functions
                    .get("MXKVStorePushPullEx")
                    .apply(new Object[] {handle, vnum, vkeys, onum, okeys, vals, outs, priority});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreSetUpdater(
            Pointer handle, MXKVStoreUpdater updater, Pointer updater_handle) {
        if (functions.containsKey("MXKVStoreSetUpdater")) {
            return functions
                    .get("MXKVStoreSetUpdater")
                    .apply(new Object[] {handle, updater, updater_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreSetUpdaterEx(
            Pointer handle,
            MXKVStoreUpdater updater,
            MXKVStoreStrUpdater str_updater,
            Pointer updater_handle) {
        if (functions.containsKey("MXKVStoreSetUpdaterEx")) {
            return functions
                    .get("MXKVStoreSetUpdaterEx")
                    .apply(new Object[] {handle, updater, str_updater, updater_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreGetType(Pointer handle, String[] type) {
        if (functions.containsKey("MXKVStoreGetType")) {
            return functions.get("MXKVStoreGetType").apply(new Object[] {handle, type});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreGetRank(Pointer handle, IntBuffer ret) {
        if (functions.containsKey("MXKVStoreGetRank")) {
            return functions.get("MXKVStoreGetRank").apply(new Object[] {handle, ret});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreGetGroupSize(Pointer handle, IntBuffer ret) {
        if (functions.containsKey("MXKVStoreGetGroupSize")) {
            return functions.get("MXKVStoreGetGroupSize").apply(new Object[] {handle, ret});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreIsWorkerNode(IntBuffer ret) {
        if (functions.containsKey("MXKVStoreIsWorkerNode")) {
            return functions.get("MXKVStoreIsWorkerNode").apply(new Object[] {ret});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreIsServerNode(IntBuffer ret) {
        if (functions.containsKey("MXKVStoreIsServerNode")) {
            return functions.get("MXKVStoreIsServerNode").apply(new Object[] {ret});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreIsSchedulerNode(IntBuffer ret) {
        if (functions.containsKey("MXKVStoreIsSchedulerNode")) {
            return functions.get("MXKVStoreIsSchedulerNode").apply(new Object[] {ret});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreBarrier(Pointer handle) {
        if (functions.containsKey("MXKVStoreBarrier")) {
            return functions.get("MXKVStoreBarrier").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreSetBarrierBeforeExit(Pointer handle, int barrier_before_exit) {
        if (functions.containsKey("MXKVStoreSetBarrierBeforeExit")) {
            return functions
                    .get("MXKVStoreSetBarrierBeforeExit")
                    .apply(new Object[] {handle, barrier_before_exit});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreRunServer(
            Pointer handle, MXKVStoreServerController controller, Pointer controller_handle) {
        if (functions.containsKey("MXKVStoreRunServer")) {
            return functions
                    .get("MXKVStoreRunServer")
                    .apply(new Object[] {handle, controller, controller_handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreSendCommmandToServers(Pointer handle, int cmd_id, String cmd_body) {
        if (functions.containsKey("MXKVStoreSendCommmandToServers")) {
            return functions
                    .get("MXKVStoreSendCommmandToServers")
                    .apply(new Object[] {handle, cmd_id, cmd_body});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreGetNumDeadNode(
            Pointer handle, int node_id, IntBuffer number, int timeout_sec) {
        if (functions.containsKey("MXKVStoreGetNumDeadNode")) {
            return functions
                    .get("MXKVStoreGetNumDeadNode")
                    .apply(new Object[] {handle, node_id, number, timeout_sec});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOWriterCreate(String uri, PointerByReference out) {
        if (functions.containsKey("MXRecordIOWriterCreate")) {
            return functions.get("MXRecordIOWriterCreate").apply(new Object[] {uri, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOWriterFree(Pointer handle) {
        if (functions.containsKey("MXRecordIOWriterFree")) {
            return functions.get("MXRecordIOWriterFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOWriterWriteRecord(Pointer handle, String buf, NativeSize size) {
        if (functions.containsKey("MXRecordIOWriterWriteRecord")) {
            return functions
                    .get("MXRecordIOWriterWriteRecord")
                    .apply(new Object[] {handle, buf, size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOWriterTell(Pointer handle, NativeSizeByReference pos) {
        if (functions.containsKey("MXRecordIOWriterTell")) {
            return functions.get("MXRecordIOWriterTell").apply(new Object[] {handle, pos});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOReaderCreate(String uri, PointerByReference out) {
        if (functions.containsKey("MXRecordIOReaderCreate")) {
            return functions.get("MXRecordIOReaderCreate").apply(new Object[] {uri, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOReaderFree(Pointer handle) {
        if (functions.containsKey("MXRecordIOReaderFree")) {
            return functions.get("MXRecordIOReaderFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOReaderReadRecord(Pointer handle, String buf, NativeSizeByReference size) {
        if (functions.containsKey("MXRecordIOReaderReadRecord")) {
            return functions
                    .get("MXRecordIOReaderReadRecord")
                    .apply(new Object[] {handle, buf, size});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOReaderSeek(Pointer handle, NativeSize pos) {
        if (functions.containsKey("MXRecordIOReaderSeek")) {
            return functions.get("MXRecordIOReaderSeek").apply(new Object[] {handle, pos});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRecordIOReaderTell(Pointer handle, NativeSizeByReference pos) {
        if (functions.containsKey("MXRecordIOReaderTell")) {
            return functions.get("MXRecordIOReaderTell").apply(new Object[] {handle, pos});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCreate(
            ByteBuffer name,
            int num_input,
            int num_output,
            PointerByReference input_names,
            PointerByReference output_names,
            PointerByReference inputs,
            PointerByReference outputs,
            ByteBuffer kernel,
            PointerByReference out) {
        if (functions.containsKey("MXRtcCreate")) {
            return functions
                    .get("MXRtcCreate")
                    .apply(
                            new Object[] {
                                name,
                                num_input,
                                num_output,
                                input_names,
                                output_names,
                                inputs,
                                outputs,
                                kernel,
                                out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcPush(
            Pointer handle,
            int num_input,
            int num_output,
            PointerByReference inputs,
            PointerByReference outputs,
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            int blockDimX,
            int blockDimY,
            int blockDimZ) {
        if (functions.containsKey("MXRtcPush")) {
            return functions
                    .get("MXRtcPush")
                    .apply(
                            new Object[] {
                                handle,
                                num_input,
                                num_output,
                                inputs,
                                outputs,
                                gridDimX,
                                gridDimY,
                                gridDimZ,
                                blockDimX,
                                blockDimY,
                                blockDimZ
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcFree(Pointer handle) {
        if (functions.containsKey("MXRtcFree")) {
            return functions.get("MXRtcFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCustomOpRegister(String op_type, CustomOpPropCreator creator) {
        if (functions.containsKey("MXCustomOpRegister")) {
            return functions.get("MXCustomOpRegister").apply(new Object[] {op_type, creator});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXCustomFunctionRecord(
            int num_inputs,
            PointerByReference inputs,
            int num_outputs,
            PointerByReference outputs,
            MXCallbackList.ByReference callbacks) {
        if (functions.containsKey("MXCustomFunctionRecord")) {
            return functions
                    .get("MXCustomFunctionRecord")
                    .apply(new Object[] {num_inputs, inputs, num_outputs, outputs, callbacks});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCudaModuleCreate(
            String source,
            int num_options,
            String[] options,
            int num_exports,
            String[] exports,
            PointerByReference out) {
        if (functions.containsKey("MXRtcCudaModuleCreate")) {
            return functions
                    .get("MXRtcCudaModuleCreate")
                    .apply(new Object[] {source, num_options, options, num_exports, exports, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCudaModuleFree(Pointer handle) {
        if (functions.containsKey("MXRtcCudaModuleFree")) {
            return functions.get("MXRtcCudaModuleFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCudaKernelCreate(
            Pointer handle,
            String name,
            int num_args,
            IntBuffer is_ndarray,
            IntBuffer is_const,
            IntBuffer arg_types,
            PointerByReference out) {
        if (functions.containsKey("MXRtcCudaKernelCreate")) {
            return functions
                    .get("MXRtcCudaKernelCreate")
                    .apply(
                            new Object[] {
                                handle, name, num_args, is_ndarray, is_const, arg_types, out
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCudaKernelFree(Pointer handle) {
        if (functions.containsKey("MXRtcCudaKernelFree")) {
            return functions.get("MXRtcCudaKernelFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXRtcCudaKernelCall(
            Pointer handle,
            int dev_id,
            PointerByReference args,
            int grid_dim_x,
            int grid_dim_y,
            int grid_dim_z,
            int block_dim_x,
            int block_dim_y,
            int block_dim_z,
            int shared_mem) {
        if (functions.containsKey("MXRtcCudaKernelCall")) {
            return functions
                    .get("MXRtcCudaKernelCall")
                    .apply(
                            new Object[] {
                                handle,
                                dev_id,
                                args,
                                grid_dim_x,
                                grid_dim_y,
                                grid_dim_z,
                                block_dim_x,
                                block_dim_y,
                                block_dim_z,
                                shared_mem
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayGetSharedMemHandle(
            Pointer handle, IntBuffer shared_pid, IntBuffer shared_id) {
        if (functions.containsKey("MXNDArrayGetSharedMemHandle")) {
            return functions
                    .get("MXNDArrayGetSharedMemHandle")
                    .apply(new Object[] {handle, shared_pid, shared_id});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateFromSharedMem(
            int shared_pid,
            int shared_id,
            int[] shape,
            int ndim,
            int dtype,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateFromSharedMem")) {
            return functions
                    .get("MXNDArrayCreateFromSharedMem")
                    .apply(new Object[] {shared_pid, shared_id, shape, ndim, dtype, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXStorageEmptyCache(int dev_type, int dev_id) {
        if (functions.containsKey("MXStorageEmptyCache")) {
            return functions.get("MXStorageEmptyCache").apply(new Object[] {dev_type, dev_id});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXNDArrayCreateFromSharedMemEx(
            int shared_pid,
            int shared_id,
            int[] shape,
            int ndim,
            int dtype,
            PointerByReference out) {
        if (functions.containsKey("MXNDArrayCreateFromSharedMemEx")) {
            return functions
                    .get("MXNDArrayCreateFromSharedMemEx")
                    .apply(new Object[] {shared_pid, shared_id, shape, ndim, dtype, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXEnginePushAsync(
            EngineAsyncFunc async_func,
            Pointer func_param,
            EngineFuncParamDeleter deleter,
            Pointer ctx_handle,
            Pointer const_vars_handle,
            int num_const_vars,
            Pointer mutable_vars_handle,
            int num_mutable_vars,
            Pointer prop_handle,
            int priority,
            String opr_name,
            byte wait) {
        if (functions.containsKey("MXEnginePushAsync")) {
            return functions
                    .get("MXEnginePushAsync")
                    .apply(
                            new Object[] {
                                async_func,
                                func_param,
                                deleter,
                                ctx_handle,
                                const_vars_handle,
                                num_const_vars,
                                mutable_vars_handle,
                                num_mutable_vars,
                                prop_handle,
                                priority,
                                opr_name,
                                wait
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXEnginePushSync(
            EngineSyncFunc sync_func,
            Pointer func_param,
            EngineFuncParamDeleter deleter,
            Pointer ctx_handle,
            Pointer const_vars_handle,
            int num_const_vars,
            Pointer mutable_vars_handle,
            int num_mutable_vars,
            Pointer prop_handle,
            int priority,
            String opr_name) {
        if (functions.containsKey("MXEnginePushSync")) {
            return functions
                    .get("MXEnginePushSync")
                    .apply(
                            new Object[] {
                                sync_func,
                                func_param,
                                deleter,
                                ctx_handle,
                                const_vars_handle,
                                num_const_vars,
                                mutable_vars_handle,
                                num_mutable_vars,
                                prop_handle,
                                priority,
                                opr_name
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXShallowCopyNDArray(Pointer src, PointerByReference out) {
        if (functions.containsKey("MXShallowCopyNDArray")) {
            return functions.get("MXShallowCopyNDArray").apply(new Object[] {src, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXShallowCopySymbol(Pointer src, PointerByReference out) {
        if (functions.containsKey("MXShallowCopySymbol")) {
            return functions.get("MXShallowCopySymbol").apply(new Object[] {src, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXEnginePushAsyncND(
            EngineAsyncFunc async_func,
            Pointer func_param,
            EngineFuncParamDeleter deleter,
            Pointer ctx_handle,
            PointerByReference const_nds_handle,
            int num_const_nds,
            PointerByReference mutable_nds_handle,
            int num_mutable_nds,
            Pointer prop_handle,
            int priority,
            String opr_name,
            byte wait) {
        if (functions.containsKey("MXEnginePushAsyncND")) {
            return functions
                    .get("MXEnginePushAsyncND")
                    .apply(
                            new Object[] {
                                async_func,
                                func_param,
                                deleter,
                                ctx_handle,
                                const_nds_handle,
                                num_const_nds,
                                mutable_nds_handle,
                                num_mutable_nds,
                                prop_handle,
                                priority,
                                opr_name,
                                wait
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXEnginePushSyncND(
            EngineSyncFunc sync_func,
            Pointer func_param,
            EngineFuncParamDeleter deleter,
            Pointer ctx_handle,
            PointerByReference const_nds_handle,
            int num_const_nds,
            PointerByReference mutable_nds_handle,
            int num_mutable_nds,
            Pointer prop_handle,
            int priority,
            String opr_name) {
        if (functions.containsKey("MXEnginePushSyncND")) {
            return functions
                    .get("MXEnginePushSyncND")
                    .apply(
                            new Object[] {
                                sync_func,
                                func_param,
                                deleter,
                                ctx_handle,
                                const_nds_handle,
                                num_const_nds,
                                mutable_nds_handle,
                                num_mutable_nds,
                                prop_handle,
                                priority,
                                opr_name
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreBroadcast(
            Pointer handle,
            int vnum,
            int[] vkeys,
            int onum,
            int[] okeys,
            PointerByReference vals,
            PointerByReference outs,
            int priority) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int MXKVStoreBroadcastEx(
            Pointer handle,
            int vnum,
            String[] vkeys,
            int onum,
            String[] okeys,
            PointerByReference vals,
            PointerByReference outs,
            int priority) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public void NNAPISetLastError(String msg) {
        if (functions.containsKey("NNAPISetLastError")) {
            functions.get("NNAPISetLastError").apply(new Object[] {msg});
        }
    }

    /** {@inheritDoc} */
    @Override
    public String NNGetLastError() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public int NNListAllOpNames(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("NNListAllOpNames")) {
            return functions.get("NNListAllOpNames").apply(new Object[] {out_size, out_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGetOpHandle(String op_name, PointerByReference op_out) {
        if (functions.containsKey("NNGetOpHandle")) {
            return functions.get("NNGetOpHandle").apply(new Object[] {op_name, op_out});
        }

        op_out.setValue(TestHelper.toPointer("This is a sample Op Pointer"));
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNListUniqueOps(IntBuffer out_size, PointerByReference out_array) {
        if (functions.containsKey("NNListUniqueOps")) {
            return functions.get("NNListUniqueOps").apply(new Object[] {out_size, out_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGetOpInfo(
            Pointer op,
            String[] real_name,
            String[] description,
            IntBuffer num_doc_args,
            PointerByReference arg_names,
            PointerByReference arg_type_infos,
            PointerByReference arg_descriptions,
            String[] return_type) {
        if (functions.containsKey("NNGetOpInfo")) {
            return functions
                    .get("NNGetOpInfo")
                    .apply(
                            new Object[] {
                                op,
                                real_name,
                                description,
                                num_doc_args,
                                arg_names,
                                arg_type_infos,
                                arg_descriptions,
                                return_type
                            });
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolCreateAtomicSymbol(
            Pointer op, int num_param, String[] keys, String[] vals, PointerByReference out) {
        if (functions.containsKey("NNSymbolCreateAtomicSymbol")) {
            return functions
                    .get("NNSymbolCreateAtomicSymbol")
                    .apply(new Object[] {op, num_param, keys, vals, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolCreateVariable(String name, PointerByReference out) {
        if (functions.containsKey("NNSymbolCreateVariable")) {
            return functions.get("NNSymbolCreateVariable").apply(new Object[] {name, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolCreateGroup(
            int num_symbols, PointerByReference symbols, PointerByReference out) {
        if (functions.containsKey("NNSymbolCreateGroup")) {
            return functions
                    .get("NNSymbolCreateGroup")
                    .apply(new Object[] {num_symbols, symbols, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNAddControlDeps(Pointer handle, Pointer src_dep) {
        if (functions.containsKey("NNAddControlDeps")) {
            return functions.get("NNAddControlDeps").apply(new Object[] {handle, src_dep});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolFree(Pointer symbol) {
        if (functions.containsKey("NNSymbolFree")) {
            return functions.get("NNSymbolFree").apply(new Object[] {symbol});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolCopy(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("NNSymbolCopy")) {
            return functions.get("NNSymbolCopy").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolPrint(Pointer symbol, String[] out_str) {
        if (functions.containsKey("NNSymbolPrint")) {
            return functions.get("NNSymbolPrint").apply(new Object[] {symbol, out_str});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolGetAttr(Pointer symbol, String key, String[] out, IntBuffer success) {
        if (functions.containsKey("NNSymbolGetAttr")) {
            return functions.get("NNSymbolGetAttr").apply(new Object[] {symbol, key, out, success});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolSetAttrs(Pointer symbol, int num_param, String[] keys, String[] values) {
        if (functions.containsKey("NNSymbolSetAttrs")) {
            return functions
                    .get("NNSymbolSetAttrs")
                    .apply(new Object[] {symbol, num_param, keys, values});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolListAttrs(
            Pointer symbol, int recursive_option, IntBuffer out_size, PointerByReference out) {
        if (functions.containsKey("NNSymbolListAttrs")) {
            return functions
                    .get("NNSymbolListAttrs")
                    .apply(new Object[] {symbol, recursive_option, out_size, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolListInputVariables(
            Pointer symbol, int option, IntBuffer out_size, PointerByReference out_sym_array) {
        if (functions.containsKey("NNSymbolListInputVariables")) {
            return functions
                    .get("NNSymbolListInputVariables")
                    .apply(new Object[] {symbol, option, out_size, out_sym_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolListInputNames(
            Pointer symbol, int option, IntBuffer out_size, PointerByReference out_str_array) {
        if (functions.containsKey("NNSymbolListInputNames")) {
            return functions
                    .get("NNSymbolListInputNames")
                    .apply(new Object[] {symbol, option, out_size, out_str_array});
        }

        out_size.put(0, 5);
        PointerArray ndarrays =
                new PointerArray(
                        TestHelper.toPointer("a"),
                        TestHelper.toPointer("b"),
                        TestHelper.toPointer("c"),
                        TestHelper.toPointer("d"),
                        TestHelper.toPointer("e"));
        out_str_array.setValue(ndarrays);
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolListOutputNames(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        if (functions.containsKey("NNSymbolListOutputNames")) {
            return functions
                    .get("NNSymbolListOutputNames")
                    .apply(new Object[] {symbol, out_size, out_str_array});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolGetNumOutputs(Pointer symbol, IntBuffer output_count) {
        if (functions.containsKey("NNSymbolGetNumOutputs")) {
            return functions
                    .get("NNSymbolGetNumOutputs")
                    .apply(new Object[] {symbol, output_count});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolGetInternals(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("NNSymbolGetInternals")) {
            return functions.get("NNSymbolGetInternals").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolGetChildren(Pointer symbol, PointerByReference out) {
        if (functions.containsKey("NNSymbolGetChildren")) {
            return functions.get("NNSymbolGetChildren").apply(new Object[] {symbol, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolGetOutput(Pointer symbol, int index, PointerByReference out) {
        if (functions.containsKey("NNSymbolGetOutput")) {
            return functions.get("NNSymbolGetOutput").apply(new Object[] {symbol, index, out});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNSymbolCompose(
            Pointer sym, String name, int num_args, String[] keys, PointerByReference args) {
        if (functions.containsKey("NNSymbolCompose")) {
            return functions
                    .get("NNSymbolCompose")
                    .apply(new Object[] {sym, name, num_args, keys, args});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphCreate(Pointer symbol, PointerByReference graph) {
        if (functions.containsKey("NNGraphCreate")) {
            return functions.get("NNGraphCreate").apply(new Object[] {symbol, graph});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphFree(Pointer handle) {
        if (functions.containsKey("NNGraphFree")) {
            return functions.get("NNGraphFree").apply(new Object[] {handle});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphGetSymbol(Pointer graph, PointerByReference symbol) {
        if (functions.containsKey("NNGraphGetSymbol")) {
            return functions.get("NNGraphGetSymbol").apply(new Object[] {graph, symbol});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphSetJSONAttr(Pointer handle, String key, String json_value) {
        if (functions.containsKey("NNGraphSetJSONAttr")) {
            return functions
                    .get("NNGraphSetJSONAttr")
                    .apply(new Object[] {handle, key, json_value});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphGetJSONAttr(
            Pointer handle, String key, String[] json_out, IntBuffer success) {
        if (functions.containsKey("NNGraphGetJSONAttr")) {
            return functions
                    .get("NNGraphGetJSONAttr")
                    .apply(new Object[] {handle, key, json_out, success});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphSetNodeEntryListAttr_(Pointer handle, String key, Pointer list) {
        if (functions.containsKey("NNGraphSetNodeEntryListAttr_")) {
            return functions
                    .get("NNGraphSetNodeEntryListAttr_")
                    .apply(new Object[] {handle, key, list});
        }
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public int NNGraphApplyPasses(
            Pointer src, int num_pass, String[] pass_names, PointerByReference dst) {
        if (functions.containsKey("NNGraphApplyPasses")) {
            return functions
                    .get("NNGraphApplyPasses")
                    .apply(new Object[] {src, num_pass, pass_names, dst});
        }
        return 0;
    }
}
// CHECKSTYLE:ON:ParameterName
