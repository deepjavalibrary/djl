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
package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;

public class MxExecutor extends NativeResource {

    private MxNDArray[] argArray;
    private MxNDArray[] auxArray;
    private MxNDArray[] dataArray;
    private MxNDArray[] outputs;
    private MxNDArray[] gradArray;

    MxExecutor(
            ResourceAllocator alloc,
            Pointer pointer,
            MxNDArray[] argArray,
            MxNDArray[] auxArray,
            MxNDArray[] dataArray,
            MxNDArray[] outputs,
            MxNDArray[] gradArray) {
        super(alloc, pointer);
        this.argArray = argArray;
        this.auxArray = auxArray;
        this.dataArray = dataArray;
        this.outputs = outputs;
        this.gradArray = gradArray;
    }

    public void forward(MxNDArray[] ndArrays, boolean forTraining) {
        for (int i = 0; i < ndArrays.length; ++i) {
            ndArrays[i].copyTo(dataArray[i]);
        }
        JnaUtils.forward(getHandle(), forTraining);
    }

    public MxNDArray[] getOutputs() {
        return outputs;
    }

    public long getExecutedBytes() {
        String[] tokens = getDebugStr().split("\n");
        String bytes = tokens[tokens.length - 2];
        tokens = bytes.split(" ");
        return Long.parseLong(tokens[1]);
    }

    public String getDebugStr() {
        return JnaUtils.getExecutorDebugString(getHandle());
    }

    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeExecutor(pointer);
        }
        if (alloc != null) {
            alloc.detach(this);
        }
        for (MxNDArray ndArray : argArray) {
            ndArray.close();
        }
        for (MxNDArray ndArray : auxArray) {
            ndArray.close();
        }
        for (MxNDArray ndArray : dataArray) {
            ndArray.close();
        }
        for (MxNDArray ndArray : outputs) {
            ndArray.close();
        }
        for (MxNDArray ndArray : gradArray) {
            if (ndArray != null) {
                ndArray.close();
            }
        }
    }
}
