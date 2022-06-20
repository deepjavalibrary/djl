/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("MissingJavadocMethod")
public final class Interpreter implements InterpreterApi {

    public static class Options extends InterpreterApi.Options {
        public Options() {}

        @Override
        public Options setNumThreads(int numThreads) {
            super.setNumThreads(numThreads);
            return this;
        }

        @Override
        public Options setUseNNAPI(boolean useNNAPI) {
            super.setUseNNAPI(useNNAPI);
            return this;
        }

        @Deprecated
        public Options setAllowFp16PrecisionForFp32(boolean allow) {
            this.allowFp16PrecisionForFp32 = allow;
            return this;
        }

        public Options addDelegate(Delegate delegate) {
            delegates.add(delegate);
            return this;
        }

        public Options setAllowBufferHandleOutput(boolean allow) {
            this.allowBufferHandleOutput = allow;
            return this;
        }

        @Override
        public Options setCancellable(boolean allow) {
            super.setCancellable(allow);
            return this;
        }

        public Options setUseXNNPACK(boolean useXNNPACK) {
            this.useXNNPACK = useXNNPACK;
            return this;
        }

        Boolean allowFp16PrecisionForFp32;
        Boolean allowBufferHandleOutput;

        // TODO(b/171856982): update the comment when applying XNNPACK delegate by default is
        // enabled for C++ TfLite library on Android platform.
        // Note: the initial "null" value indicates default behavior which may mean XNNPACK
        // delegate will be applied by default.
        Boolean useXNNPACK;
        final List<Delegate> delegates = new ArrayList<>();
    }

    public Interpreter(File modelFile) {
        this(modelFile, /*options = */ null);
    }

    public Interpreter(File modelFile, Options options) {
        wrapper = new NativeInterpreterWrapper(modelFile.getAbsolutePath(), options);
        signatureNameList = getSignatureDefNames();
    }

    public Interpreter(ByteBuffer byteBuffer) {
        this(byteBuffer, /* options= */ null);
    }

    public Interpreter(ByteBuffer byteBuffer, Options options) {
        wrapper = new NativeInterpreterWrapper(byteBuffer, options);
        signatureNameList = getSignatureDefNames();
    }

    @Override
    public void run(Object input, Object output) {
        Object[] inputs = {input};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, output);
        runForMultipleInputsOutputs(inputs, outputs);
    }

    @Override
    public void runForMultipleInputsOutputs(Object[] inputs, Map<Integer, Object> outputs) {
        checkNotClosed();
        wrapper.run(inputs, outputs);
    }

    public void runSignature(
            Map<String, Object> inputs, Map<String, Object> outputs, String methodName) {
        checkNotClosed();
        if (methodName == null && signatureNameList.length == 1) {
            methodName = signatureNameList[0];
        }
        if (methodName == null) {
            throw new IllegalArgumentException(
                    "Input error: SignatureDef methodName should not be null. null is only allowed"
                            + " if the model has a single Signature. Available Signatures: "
                            + Arrays.toString(signatureNameList));
        }
        wrapper.runSignature(inputs, outputs, methodName);
    }

    public void runSignature(Map<String, Object> inputs, Map<String, Object> outputs) {
        checkNotClosed();
        runSignature(inputs, outputs, null);
    }

    @Override
    public void allocateTensors() {
        checkNotClosed();
        wrapper.allocateTensors();
    }

    @Override
    public void resizeInput(int idx, int[] dims) {
        checkNotClosed();
        wrapper.resizeInput(idx, dims, false);
    }

    @Override
    public void resizeInput(int idx, int[] dims, boolean strict) {
        checkNotClosed();
        wrapper.resizeInput(idx, dims, strict);
    }

    @Override
    public int getInputTensorCount() {
        checkNotClosed();
        return wrapper.getInputTensorCount();
    }

    @Override
    public int getInputIndex(String opName) {
        checkNotClosed();
        return wrapper.getInputIndex(opName);
    }

    @Override
    public Tensor getInputTensor(int inputIndex) {
        checkNotClosed();
        return wrapper.getInputTensor(inputIndex);
    }

    public Tensor getInputTensorFromSignature(String inputName, String methodName) {
        checkNotClosed();
        if (methodName == null && signatureNameList.length == 1) {
            methodName = signatureNameList[0];
        }
        if (methodName == null) {
            throw new IllegalArgumentException(
                    "Input error: SignatureDef methodName should not be null. null is only allowed"
                            + " if the model has a single Signature. Available Signatures: "
                            + Arrays.toString(signatureNameList));
        }
        return wrapper.getInputTensor(inputName, methodName);
    }

    public String[] getSignatureDefNames() {
        checkNotClosed();
        return wrapper.getSignatureDefNames();
    }

    public String[] getSignatureInputs(String methodName) {
        checkNotClosed();
        return wrapper.getSignatureInputs(methodName);
    }

    public String[] getSignatureOutputs(String methodName) {
        checkNotClosed();
        return wrapper.getSignatureOutputs(methodName);
    }

    @Override
    public int getOutputTensorCount() {
        checkNotClosed();
        return wrapper.getOutputTensorCount();
    }

    @Override
    public int getOutputIndex(String opName) {
        checkNotClosed();
        return wrapper.getOutputIndex(opName);
    }

    @Override
    public Tensor getOutputTensor(int outputIndex) {
        checkNotClosed();
        return wrapper.getOutputTensor(outputIndex);
    }

    public Tensor getOutputTensorFromSignature(String outputName, String methodName) {
        checkNotClosed();
        if (methodName == null && signatureNameList.length == 1) {
            methodName = signatureNameList[0];
        }
        if (methodName == null) {
            throw new IllegalArgumentException(
                    "Input error: SignatureDef methodName should not be null. null is only allowed"
                            + " if the model has a single Signature. Available Signatures: "
                            + Arrays.toString(signatureNameList));
        }
        return wrapper.getOutputTensor(outputName, methodName);
    }

    @Override
    public Long getLastNativeInferenceDurationNanoseconds() {
        checkNotClosed();
        return wrapper.getLastNativeInferenceDurationNanoseconds();
    }

    public void resetVariableTensors() {
        checkNotClosed();
        wrapper.resetVariableTensors();
    }

    public void setCancelled(boolean cancelled) {
        wrapper.setCancelled(cancelled);
    }

    int getExecutionPlanLength() {
        checkNotClosed();
        return wrapper.getExecutionPlanLength();
    }

    @Override
    public void close() {
        if (wrapper != null) {
            wrapper.close();
            wrapper = null;
        }
    }

    // for Object.finalize, see https://bugs.openjdk.java.net/browse/JDK-8165641
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    private void checkNotClosed() {
        if (wrapper == null) {
            throw new IllegalStateException(
                    "Internal error: The Interpreter has already been closed.");
        }
    }

    NativeInterpreterWrapper wrapper;
    String[] signatureNameList;
}
