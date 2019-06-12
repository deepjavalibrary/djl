package com.amazon.ai.test;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.mxnet.jna.LibFeature;
import org.apache.mxnet.jna.MXCallbackList;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.jna.NativeSize;
import org.apache.mxnet.jna.NativeSizeByReference;
import org.apache.mxnet.jna.PointerArray;

// CHECKSTYLE:OFF:ParameterName
public class MockMxnetLibrary implements MxnetLibrary {

    private Map<String, Function<Object[], Integer>> functions = new HashMap<>();

    public void setFunction(String funcName, Function<Object[], Integer> func) {
        functions.put(funcName, func);
    }

    public void resetFunctions() {
        functions = new HashMap<>();
    }

    @Override
    public String MXGetLastError() {
        return null;
    }

    @Override
    public int MXLibInfoFeatures(LibFeature.ByReference[] libFeature, NativeSizeByReference size) {
        return 0;
    }

    @Override
    public int MXRandomSeed(int seed) {
        return 0;
    }

    @Override
    public int MXRandomSeedContext(int seed, int dev_type, int dev_id) {
        return 0;
    }

    @Override
    public int MXNotifyShutdown() {
        return 0;
    }

    @Override
    public int MXSetProcessProfilerConfig(
            int num_params, String[] keys, String[] vals, Pointer kvstoreHandle) {
        return 0;
    }

    @Override
    public int MXSetProfilerConfig(int num_params, String[] keys, String[] vals) {
        return 0;
    }

    @Override
    public int MXSetProcessProfilerState(int state, int profile_process, Pointer kvStoreHandle) {
        return 0;
    }

    @Override
    public int MXSetProfilerState(int state) {
        return 0;
    }

    @Override
    public int MXDumpProcessProfile(int finished, int profile_process, Pointer kvStoreHandle) {
        return 0;
    }

    @Override
    public int MXDumpProfile(int finished) {
        return 0;
    }

    @Override
    public int MXAggregateProfileStatsPrint(String[] out_str, int reset) {
        return 0;
    }

    @Override
    public int MXProcessProfilePause(int paused, int profile_process, Pointer kvStoreHandle) {
        return 0;
    }

    @Override
    public int MXProfilePause(int paused) {
        return 0;
    }

    @Override
    public int MXProfileCreateDomain(String domain, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXProfileCreateTask(Pointer domain, String task_name, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXProfileCreateFrame(Pointer domain, String frame_name, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXProfileCreateEvent(String event_name, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXProfileCreateCounter(Pointer domain, String counter_name, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXProfileDestroyHandle(Pointer frame_handle) {
        return 0;
    }

    @Override
    public int MXProfileDurationStart(Pointer duration_handle) {
        return 0;
    }

    @Override
    public int MXProfileDurationStop(Pointer duration_handle) {
        return 0;
    }

    @Override
    public int MXProfileSetCounter(Pointer counter_handle, long value) {
        return 0;
    }

    @Override
    public int MXProfileAdjustCounter(Pointer counter_handle, long value) {
        return 0;
    }

    @Override
    public int MXProfileSetMarker(Pointer domain, String instant_marker_name, String scope) {
        return 0;
    }

    @Override
    public int MXSetNumOMPThreads(int thread_num) {
        return 0;
    }

    @Override
    public int MXEngineSetBulkSize(int bulk_size, IntBuffer prev_bulk_size) {
        return 0;
    }

    @Override
    public int MXGetGPUCount(IntBuffer out) {
        out.put(0, 1);
        return 0;
    }

    @Override
    public int MXGetGPUMemoryInformation(int dev, IntBuffer free_mem, IntBuffer total_mem) {
        return 0;
    }

    @Override
    public int MXGetGPUMemoryInformation64(int dev, LongBuffer free_mem, LongBuffer total_mem) {
        free_mem.put(900);
        total_mem.put(1000);
        return 0;
    }

    @Override
    public int MXGetVersion(IntBuffer out) {
        out.put(0, 10500);
        return 0;
    }

    @Override
    public int MXNDArrayCreateNone(PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayCreate(
            int[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayCreateEx(
            int[] shape,
            int ndim,
            int dev_type,
            int dev_id,
            int delay_alloc,
            int dtype,
            PointerByReference out) {
        out.setValue(new PointerArray());
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXNDArrayLoadFromRawBytes(Pointer buf, NativeSize size, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArraySaveRawBytes(
            Pointer handle, NativeSizeByReference out_size, PointerByReference out_buf) {
        return 0;
    }

    @Override
    public int MXNDArraySave(String fname, int num_args, PointerArray args, String[] keys) {
        return 0;
    }

    @Override
    public int MXNDArrayLoad(
            String fname,
            IntBuffer out_size,
            PointerByReference out_arr,
            IntBuffer out_name_size,
            PointerByReference out_names) {
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

    @Override
    public int MXNDArrayLoadFromBuffer(
            Pointer ndarray_buffer,
            NativeSize size,
            IntBuffer out_size,
            PointerByReference out_arr,
            IntBuffer out_name_size,
            PointerByReference out_names) {
        return 0;
    }

    @Override
    public int MXNDArraySyncCopyFromCPU(Pointer handle, Pointer data, NativeSize size) {
        if (functions.containsKey("MXNDArraySyncCopyFromCPU")) {
            return functions
                    .get("MXNDArraySyncCopyFromCPU")
                    .apply(new Object[] {handle, data, size});
        }
        return 0;
    }

    @Override
    public int MXNDArraySyncCopyToCPU(Pointer handle, Pointer data, NativeSize size) {
        return 0;
    }

    @Override
    public int MXNDArraySyncCopyFromNDArray(Pointer handle_dst, Pointer handle_src, int i) {
        return 0;
    }

    @Override
    public int MXNDArraySyncCheckFormat(Pointer handle, byte full_check) {
        return 0;
    }

    @Override
    public int MXNDArrayWaitToRead(Pointer handle) {
        return 0;
    }

    @Override
    public int MXNDArrayWaitToWrite(Pointer handle) {
        return 0;
    }

    @Override
    public int MXNDArrayWaitAll() {
        return 0;
    }

    @Override
    public int MXNDArrayFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXNDArraySlice(
            Pointer handle, int slice_begin, int slice_end, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayAt(Pointer handle, int idx, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayGetStorageType(Pointer handle, IntBuffer out_storage_type) {
        out_storage_type.put(0, 2);
        return 0;
    }

    @Override
    public int MXNDArrayReshape(Pointer handle, int ndim, IntBuffer dims, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayReshape64(
            Pointer handle, int ndim, LongBuffer dims, byte reverse, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayGetShape(Pointer handle, IntBuffer out_dim, PointerByReference out_pdata) {
        out_dim.put(0, 3);
        Pointer ptr = TestHelper.toPointer(new int[] {1, 2, 3});
        out_pdata.setValue(ptr);
        return 0;
    }

    @Override
    public int MXNDArrayGetShapeEx(
            Pointer handle, IntBuffer out_dim, PointerByReference out_pdata) {
        return 0;
    }

    @Override
    public int MXNDArrayGetData(Pointer handle, PointerByReference out_pdata) {
        return 0;
    }

    @Override
    public int MXNDArrayToDLPack(Pointer handle, PointerByReference out_dlpack) {
        return 0;
    }

    @Override
    public int MXNDArrayFromDLPack(
            Pointer dlpack, byte transient_handle, PointerByReference out_handle) {
        return 0;
    }

    @Override
    public int MXNDArrayCallDLPackDeleter(Pointer dlpack) {
        return 0;
    }

    @Override
    public int MXNDArrayGetDType(Pointer handle, IntBuffer out_dtype) {
        return 0;
    }

    @Override
    public int MXNDArrayGetAuxType(Pointer handle, int i, IntBuffer out_type) {
        return 0;
    }

    @Override
    public int MXNDArrayGetAuxNDArray(Pointer handle, int i, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayGetDataNDArray(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayGetContext(Pointer handle, IntBuffer out_dev_type, IntBuffer out_dev_id) {
        out_dev_type.put(0, 2);
        out_dev_id.put(1);
        return 0;
    }

    @Override
    public int MXNDArrayGetGrad(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArrayDetach(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXNDArraySetGradState(Pointer handle, int state) {
        return 0;
    }

    @Override
    public int MXNDArrayGetGradState(Pointer handle, IntBuffer out) {
        return 0;
    }

    @Override
    public int MXListFunctions(IntBuffer out_size, PointerByReference out_array) {
        return 0;
    }

    @Override
    public int MXGetFunction(String name, PointerByReference out) {
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXFuncDescribe(
            Pointer fun,
            IntBuffer num_use_vars,
            IntBuffer num_scalars,
            IntBuffer num_mutate_vars,
            IntBuffer type_mask) {
        return 0;
    }

    @Override
    public int MXFuncInvoke(
            Pointer fun,
            PointerByReference use_vars,
            FloatBuffer scalar_args,
            PointerByReference mutate_vars) {
        return 0;
    }

    @Override
    public int MXFuncInvokeEx(
            Pointer fun,
            PointerByReference use_vars,
            FloatBuffer scalar_args,
            PointerByReference mutate_vars,
            int num_params,
            PointerByReference param_keys,
            PointerByReference param_vals) {
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXImperativeInvokeEx(
            Pointer creator,
            int num_inputs,
            PointerByReference inputs,
            IntBuffer num_outputs,
            PointerByReference outputs,
            int num_params,
            String[] param_keys,
            String[] param_vals,
            PointerByReference out_stypes) {
        return 0;
    }

    @Override
    public int MXAutogradSetIsRecording(int is_recording, IntBuffer prev) {
        return 0;
    }

    @Override
    public int MXAutogradSetIsTraining(int is_training, IntBuffer prev) {
        return 0;
    }

    @Override
    public int MXAutogradIsRecording(ByteBuffer curr) {
        return 0;
    }

    @Override
    public int MXAutogradIsTraining(ByteBuffer curr) {
        return 0;
    }

    @Override
    public int MXIsNumpyShape(ByteBuffer curr) {
        return 0;
    }

    @Override
    public int MXSetIsNumpyShape(int is_np_shape, IntBuffer prev) {
        return 0;
    }

    @Override
    public int MXAutogradMarkVariables(
            int num_var,
            PointerByReference var_handles,
            IntBuffer reqs_array,
            PointerByReference grad_handles) {
        return 0;
    }

    @Override
    public int MXAutogradComputeGradient(int num_output, PointerByReference output_handles) {
        return 0;
    }

    @Override
    public int MXAutogradBackward(
            int num_output,
            PointerByReference output_handles,
            PointerByReference ograd_handles,
            int retain_graph) {
        return 0;
    }

    @Override
    public int MXAutogradBackwardEx(
            int num_output,
            PointerByReference output_handles,
            PointerByReference ograd_handles,
            int num_variables,
            PointerByReference var_handles,
            int retain_graph,
            int create_graph,
            int is_train,
            PointerByReference grad_handles,
            PointerByReference grad_stypes) {
        return 0;
    }

    @Override
    public int MXAutogradGetSymbol(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXCreateCachedOp(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXCreateCachedOpEx(
            Pointer handle, int num_flags, String[] keys, String[] vals, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXFreeCachedOp(Pointer handle) {
        return 0;
    }

    @Override
    public int MXInvokeCachedOp(
            Pointer handle,
            int num_inputs,
            Pointer inputs,
            IntBuffer num_outputs,
            PointerByReference outputs) {
        num_outputs.put(0, 3);
        outputs.setValue(new PointerArray());
        return 0;
    }

    @Override
    public int MXInvokeCachedOpEx(
            Pointer handle,
            int num_inputs,
            PointerByReference inputs,
            IntBuffer num_outputs,
            PointerByReference outputs,
            PointerByReference out_stypes) {
        return 0;
    }

    @Override
    public int MXListAllOpNames(IntBuffer out_size, PointerByReference out_array) {
        PointerArray pa = new PointerArray(TestHelper.toPointer("softmax"));
        out_size.put(0, 1);
        out_array.setValue(pa);
        return 0;
    }

    @Override
    public int MXSymbolListAtomicSymbolCreators(IntBuffer out_size, PointerByReference out_array) {
        return 0;
    }

    @Override
    public int MXSymbolGetAtomicSymbolName(Pointer creator, String[] name) {
        return 0;
    }

    @Override
    public int MXSymbolGetInputSymbols(
            Pointer sym, PointerByReference inputs, IntBuffer input_size) {
        return 0;
    }

    @Override
    public int MXSymbolCutSubgraph(Pointer sym, PointerByReference inputs, IntBuffer input_size) {
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXSymbolCreateAtomicSymbol(
            Pointer creator, int num_param, String[] keys, String[] vals, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolCreateVariable(String name, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolCreateGroup(
            int num_symbols, PointerByReference symbols, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolCreateFromFile(String fname, PointerByReference out) {
        out.setValue(new PointerArray());
        return 0;
    }

    @Override
    public int MXSymbolCreateFromJSON(String json, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolRemoveAmpCast(Pointer sym_handle, PointerByReference ret_sym_handle) {
        return 0;
    }

    @Override
    public int MXSymbolSaveToFile(Pointer symbol, String fname) {
        return 0;
    }

    @Override
    public int MXSymbolSaveToJSON(Pointer symbol, String[] out_json) {
        return 0;
    }

    @Override
    public int MXSymbolFree(Pointer symbol) {
        return 0;
    }

    @Override
    public int MXSymbolCopy(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolPrint(Pointer symbol, String[] out_str) {
        return 0;
    }

    @Override
    public int MXSymbolGetName(Pointer symbol, String[] out, IntBuffer success) {
        return 0;
    }

    @Override
    public int MXSymbolGetAttr(Pointer symbol, String key, String[] out, IntBuffer success) {
        return 0;
    }

    @Override
    public int MXSymbolSetAttr(Pointer symbol, String key, String value) {
        return 0;
    }

    @Override
    public int MXSymbolListAttr(Pointer symbol, IntBuffer out_size, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolListAttrShallow(Pointer symbol, IntBuffer out_size, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolListArguments(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        return 0;
    }

    @Override
    public int MXSymbolListOutputs(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {

        return 0;
    }

    @Override
    public int MXSymbolGetNumOutputs(Pointer symbol, IntBuffer output_count) {
        return 0;
    }

    @Override
    public int MXSymbolGetInternals(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolGetChildren(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolGetOutput(Pointer symbol, int index, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXSymbolListAuxiliaryStates(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        return 0;
    }

    @Override
    public int MXSymbolCompose(
            Pointer sym, String name, int num_args, String[] keys, PointerByReference args) {
        return 0;
    }

    @Override
    public int MXSymbolGrad(Pointer sym, int num_wrt, String[] wrt, PointerByReference out) {
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXQuantizeSymbol(
            Pointer sym_handle,
            PointerByReference ret_sym_handle,
            int num_excluded_symbols,
            String[] excluded_symbols,
            int num_offline,
            String[] offline_params,
            String quantized_dtype,
            byte calib_quantize) {
        return 0;
    }

    @Override
    public int MXSetCalibTableToQuantizedSymbol(
            Pointer qsym_handle,
            int num_layers,
            String[] layer_names,
            FloatBuffer low_quantiles,
            FloatBuffer high_quantiles,
            PointerByReference ret_sym_handle) {
        return 0;
    }

    @Override
    public int MXGenBackendSubgraph(
            Pointer sym_handle, String backend, PointerByReference ret_sym_handle) {
        return 0;
    }

    @Override
    public int MXGenAtomicSymbolFromSymbol(Pointer sym_handle, PointerByReference ret_sym_handle) {
        return 0;
    }

    @Override
    public int MXExecutorFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXExecutorPrint(Pointer handle, String[] out_str) {
        return 0;
    }

    @Override
    public int MXExecutorForward(Pointer handle, int is_train) {
        return 0;
    }

    @Override
    public int MXExecutorBackward(Pointer handle, int len, PointerByReference head_grads) {
        return 0;
    }

    @Override
    public int MXExecutorBackwardEx(
            Pointer handle, int len, PointerByReference head_grads, int is_train) {
        return 0;
    }

    @Override
    public int MXExecutorOutputs(Pointer handle, IntBuffer out_size, PointerByReference out) {
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXExecutorGetOptimizedSymbol(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXExecutorSetMonitorCallback(
            Pointer handle, ExecutorMonitorCallback callback, Pointer callback_handle) {
        return 0;
    }

    @Override
    public int MXExecutorSetMonitorCallbackEX(
            Pointer handle,
            ExecutorMonitorCallback callback,
            Pointer callback_handle,
            byte monitor_all) {
        return 0;
    }

    @Override
    public int MXListDataIters(IntBuffer out_size, PointerByReference out_array) {
        return 0;
    }

    @Override
    public int MXDataIterCreateIter(
            Pointer handle, int num_param, String[] keys, String[] vals, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXDataIterGetIterInfo(
            Pointer creator,
            String[] name,
            String[] description,
            IntBuffer num_args,
            PointerByReference arg_names,
            PointerByReference arg_type_infos,
            PointerByReference arg_descriptions) {
        return 0;
    }

    @Override
    public int MXDataIterFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXDataIterNext(Pointer handle, IntBuffer out) {
        return 0;
    }

    @Override
    public int MXDataIterBeforeFirst(Pointer handle) {
        return 0;
    }

    @Override
    public int MXDataIterGetData(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXDataIterGetIndex(
            Pointer handle, PointerByReference out_index, LongBuffer out_size) {
        return 0;
    }

    @Override
    public int MXDataIterGetPadNum(Pointer handle, IntBuffer pad) {
        return 0;
    }

    @Override
    public int MXDataIterGetLabel(Pointer handle, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXInitPSEnv(int num_vars, String[] keys, String[] vals) {
        return 0;
    }

    @Override
    public int MXKVStoreCreate(String type, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXKVStoreSetGradientCompression(
            Pointer handle, int num_params, String[] keys, String[] vals) {
        return 0;
    }

    @Override
    public int MXKVStoreFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXKVStoreInit(Pointer handle, int num, int[] keys, PointerByReference vals) {
        return 0;
    }

    @Override
    public int MXKVStoreInitEx(Pointer handle, int num, String[] keys, PointerByReference vals) {
        return 0;
    }

    @Override
    public int MXKVStorePush(
            Pointer handle, int num, int[] keys, PointerByReference vals, int priority) {
        return 0;
    }

    @Override
    public int MXKVStorePushEx(
            Pointer handle, int num, String[] keys, PointerByReference vals, int priority) {
        return 0;
    }

    @Override
    public int MXKVStorePullWithSparse(
            Pointer handle,
            int num,
            int[] keys,
            PointerByReference vals,
            int priority,
            byte ignore_sparse) {
        return 0;
    }

    @Override
    public int MXKVStorePullWithSparseEx(
            Pointer handle,
            int num,
            String[] keys,
            PointerByReference vals,
            int priority,
            byte ignore_sparse) {
        return 0;
    }

    @Override
    public int MXKVStorePull(
            Pointer handle, int num, int[] keys, PointerByReference vals, int priority) {
        return 0;
    }

    @Override
    public int MXKVStorePullEx(
            Pointer handle, int num, String[] keys, PointerByReference vals, int priority) {
        return 0;
    }

    @Override
    public int MXKVStorePullRowSparse(
            Pointer handle,
            int num,
            int[] keys,
            PointerByReference vals,
            PointerByReference row_ids,
            int priority) {
        return 0;
    }

    @Override
    public int MXKVStorePullRowSparseEx(
            Pointer handle,
            int num,
            String[] keys,
            PointerByReference vals,
            PointerByReference row_ids,
            int priority) {
        return 0;
    }

    @Override
    public int MXKVStoreSetUpdater(
            Pointer handle, MXKVStoreUpdater updater, Pointer updater_handle) {
        return 0;
    }

    @Override
    public int MXKVStoreSetUpdaterEx(
            Pointer handle,
            MXKVStoreUpdater updater,
            MXKVStoreStrUpdater str_updater,
            Pointer updater_handle) {
        return 0;
    }

    @Override
    public int MXKVStoreGetType(Pointer handle, String[] type) {
        return 0;
    }

    @Override
    public int MXKVStoreGetRank(Pointer handle, IntBuffer ret) {
        return 0;
    }

    @Override
    public int MXKVStoreGetGroupSize(Pointer handle, IntBuffer ret) {
        return 0;
    }

    @Override
    public int MXKVStoreIsWorkerNode(IntBuffer ret) {
        return 0;
    }

    @Override
    public int MXKVStoreIsServerNode(IntBuffer ret) {
        return 0;
    }

    @Override
    public int MXKVStoreIsSchedulerNode(IntBuffer ret) {
        return 0;
    }

    @Override
    public int MXKVStoreBarrier(Pointer handle) {
        return 0;
    }

    @Override
    public int MXKVStoreSetBarrierBeforeExit(Pointer handle, int barrier_before_exit) {
        return 0;
    }

    @Override
    public int MXKVStoreRunServer(
            Pointer handle, MXKVStoreServerController controller, Pointer controller_handle) {
        return 0;
    }

    @Override
    public int MXKVStoreSendCommmandToServers(Pointer handle, int cmd_id, String cmd_body) {
        return 0;
    }

    @Override
    public int MXKVStoreGetNumDeadNode(
            Pointer handle, int node_id, IntBuffer number, int timeout_sec) {
        return 0;
    }

    @Override
    public int MXRecordIOWriterCreate(String uri, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXRecordIOWriterFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXRecordIOWriterWriteRecord(Pointer handle, String buf, NativeSize size) {
        return 0;
    }

    @Override
    public int MXRecordIOWriterTell(Pointer handle, NativeSizeByReference pos) {
        return 0;
    }

    @Override
    public int MXRecordIOReaderCreate(String uri, PointerByReference out) {
        return 0;
    }

    @Override
    public int MXRecordIOReaderFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXRecordIOReaderReadRecord(Pointer handle, String buf, NativeSizeByReference size) {
        return 0;
    }

    @Override
    public int MXRecordIOReaderSeek(Pointer handle, NativeSize pos) {
        return 0;
    }

    @Override
    public int MXRecordIOReaderTell(Pointer handle, NativeSizeByReference pos) {
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXRtcFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXCustomOpRegister(String op_type, CustomOpPropCreator creator) {
        return 0;
    }

    @Override
    public int MXCustomFunctionRecord(
            int num_inputs,
            PointerByReference inputs,
            int num_outputs,
            PointerByReference outputs,
            MXCallbackList.ByReference callbacks) {
        return 0;
    }

    @Override
    public int MXRtcCudaModuleCreate(
            String source,
            int num_options,
            String[] options,
            int num_exports,
            String[] exports,
            PointerByReference out) {
        return 0;
    }

    @Override
    public int MXRtcCudaModuleFree(Pointer handle) {
        return 0;
    }

    @Override
    public int MXRtcCudaKernelCreate(
            Pointer handle,
            String name,
            int num_args,
            IntBuffer is_ndarray,
            IntBuffer is_const,
            IntBuffer arg_types,
            PointerByReference out) {
        return 0;
    }

    @Override
    public int MXRtcCudaKernelFree(Pointer handle) {
        return 0;
    }

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
        return 0;
    }

    @Override
    public int MXNDArrayGetSharedMemHandle(
            Pointer handle, IntBuffer shared_pid, IntBuffer shared_id) {
        return 0;
    }

    @Override
    public int MXNDArrayCreateFromSharedMem(
            int shared_pid,
            int shared_id,
            int[] shape,
            int ndim,
            int dtype,
            PointerByReference out) {
        return 0;
    }

    @Override
    public int MXStorageEmptyCache(int dev_type, int dev_id) {
        return 0;
    }

    @Override
    public int MXNDArrayCreateFromSharedMemEx(
            int shared_pid,
            int shared_id,
            int[] shape,
            int ndim,
            int dtype,
            PointerByReference out) {
        return 0;
    }

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
        return 0;
    }

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
        return 0;
    }

    @Override
    public void NNAPISetLastError(String msg) {}

    @Override
    public String NNGetLastError() {
        return null;
    }

    @Override
    public int NNListAllOpNames(IntBuffer out_size, PointerByReference out_array) {
        return 0;
    }

    @Override
    public int NNGetOpHandle(String op_name, PointerByReference op_out) {
        return 0;
    }

    @Override
    public int NNListUniqueOps(IntBuffer out_size, PointerByReference out_array) {
        return 0;
    }

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
        return 0;
    }

    @Override
    public int NNSymbolCreateAtomicSymbol(
            Pointer op, int num_param, String[] keys, String[] vals, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolCreateVariable(String name, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolCreateGroup(
            int num_symbols, PointerByReference symbols, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNAddControlDeps(Pointer handle, Pointer src_dep) {
        return 0;
    }

    @Override
    public int NNSymbolFree(Pointer symbol) {
        return 0;
    }

    @Override
    public int NNSymbolCopy(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolPrint(Pointer symbol, String[] out_str) {
        return 0;
    }

    @Override
    public int NNSymbolGetAttr(Pointer symbol, String key, String[] out, IntBuffer success) {
        return 0;
    }

    @Override
    public int NNSymbolSetAttrs(Pointer symbol, int num_param, String[] keys, String[] values) {
        return 0;
    }

    @Override
    public int NNSymbolListAttrs(
            Pointer symbol, int recursive_option, IntBuffer out_size, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolListInputVariables(
            Pointer symbol, int option, IntBuffer out_size, PointerByReference out_sym_array) {
        return 0;
    }

    @Override
    public int NNSymbolListInputNames(
            Pointer symbol, int option, IntBuffer out_size, PointerByReference out_str_array) {
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

    @Override
    public int NNSymbolListOutputNames(
            Pointer symbol, IntBuffer out_size, PointerByReference out_str_array) {
        return 0;
    }

    @Override
    public int NNSymbolGetNumOutputs(Pointer symbol, IntBuffer output_count) {
        return 0;
    }

    @Override
    public int NNSymbolGetInternals(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolGetChildren(Pointer symbol, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolGetOutput(Pointer symbol, int index, PointerByReference out) {
        return 0;
    }

    @Override
    public int NNSymbolCompose(
            Pointer sym, String name, int num_args, String[] keys, PointerByReference args) {
        return 0;
    }

    @Override
    public int NNGraphCreate(Pointer symbol, PointerByReference graph) {
        return 0;
    }

    @Override
    public int NNGraphFree(Pointer handle) {
        return 0;
    }

    @Override
    public int NNGraphGetSymbol(Pointer graph, PointerByReference symbol) {
        return 0;
    }

    @Override
    public int NNGraphSetJSONAttr(Pointer handle, String key, String json_value) {
        return 0;
    }

    @Override
    public int NNGraphGetJSONAttr(
            Pointer handle, String key, String[] json_out, IntBuffer success) {
        return 0;
    }

    @Override
    public int NNGraphSetNodeEntryListAttr_(Pointer handle, String key, Pointer list) {
        return 0;
    }

    @Override
    public int NNGraphApplyPasses(
            Pointer src, int num_pass, String[] pass_names, PointerByReference dst) {
        return 0;
    }
}
// CHECKSTYLE:ON:ParameterName
