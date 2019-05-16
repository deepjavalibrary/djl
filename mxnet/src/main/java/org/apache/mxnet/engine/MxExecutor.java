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

import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;

public class MxExecutor extends NativeResource {

    private NDArray[] argArray;
    private NDArray[] auxArray;
    private NDArray[] dataArray;
    private NDArray[] outputs;
    private NDArray[] gradArray;

    MxExecutor(
            Pointer pointer,
            NDArray[] argArray,
            NDArray[] auxArray,
            NDArray[] dataArray,
            NDArray[] outputs,
            NDArray[] gradArray) {
        super(pointer);
        this.argArray = argArray;
        this.auxArray = auxArray;
        this.dataArray = dataArray;
        this.outputs = outputs;
        this.gradArray = gradArray;
    }

    public void forward(NDList ndList, boolean forTraining) {
        int i = 0;
        for (NDArray array : ndList) {
            array.copyTo(dataArray[i++]);
        }
        JnaUtils.forward(getHandle(), forTraining);
    }

    public NDList getOutputs() {
        return new NDList(outputs);
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
        for (NDArray ndArray : argArray) {
            ndArray.close();
        }
        for (NDArray ndArray : auxArray) {
            ndArray.close();
        }

        // dataArray is just a pointer to argArray, no need to close

        for (NDArray ndArray : outputs) {
            ndArray.close();
        }
        for (NDArray ndArray : gradArray) {
            if (ndArray != null) {
                ndArray.close();
            }
        }
    }
}
