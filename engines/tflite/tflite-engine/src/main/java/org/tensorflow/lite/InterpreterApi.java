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

import java.util.Map;

@SuppressWarnings("MissingJavadocMethod")
public interface InterpreterApi extends AutoCloseable {

    public static class Options {
        public Options() {}

        public Options setNumThreads(int numThreads) {
            this.numThreads = numThreads;
            return this;
        }

        public Options setUseNNAPI(boolean useNNAPI) {
            this.useNNAPI = useNNAPI;
            return this;
        }

        public Options setCancellable(boolean allow) {
            this.allowCancellation = allow;
            return this;
        }

        int numThreads = -1;
        Boolean useNNAPI;
        Boolean allowCancellation;
    }

    public void run(Object input, Object output);

    public void runForMultipleInputsOutputs(Object[] inputs, Map<Integer, Object> outputs);

    public void allocateTensors();

    public void resizeInput(int idx, int[] dims);

    public void resizeInput(int idx, int[] dims, boolean strict);

    public int getInputTensorCount();

    public int getInputIndex(String opName);

    public Tensor getInputTensor(int inputIndex);

    public int getOutputTensorCount();

    public int getOutputIndex(String opName);

    public Tensor getOutputTensor(int outputIndex);

    public Long getLastNativeInferenceDurationNanoseconds();

    @Override
    void close();
}
