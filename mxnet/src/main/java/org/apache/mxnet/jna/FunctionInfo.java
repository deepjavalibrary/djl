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
package org.apache.mxnet.jna;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDScopedFactory;
import software.amazon.ai.util.PairList;

public class FunctionInfo {

    private Pointer handle;
    private String name;
    private PairList<String, String> arguments;

    FunctionInfo(Pointer pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    public int invoke(
            NDScopedFactory factory, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        Pointer[] handles =
                Arrays.stream(src).map(a -> ((MxNDArray) a).getHandle()).toArray(Pointer[]::new);
        PointerArray srcHandles = new PointerArray(handles);
        handles = Arrays.stream(dest).map(a -> ((MxNDArray) a).getHandle()).toArray(Pointer[]::new);
        PointerByReference destRef = new PointerByReference(new PointerArray(handles));
        return JnaUtils.imperativeInvoke(handle, srcHandles, destRef, params);
    }

    public NDArray[] invoke(NDScopedFactory factory, NDArray[] src, PairList<String, ?> params) {
        Pointer[] handles =
                Arrays.stream(src).map(a -> ((MxNDArray) a).getHandle()).toArray(Pointer[]::new);
        PointerArray srcHandles = new PointerArray(handles);
        return invoke((MxNDFactory) factory, srcHandles, params);
    }

    public NDArray[] invoke(NDScopedFactory factory, NDArray src, PairList<String, ?> params) {
        PointerArray handles = new PointerArray(((MxNDArray) src).getHandle());
        return invoke((MxNDFactory) factory, handles, params);
    }

    private NDArray[] invoke(MxNDFactory factory, PointerArray src, PairList<String, ?> params) {
        PointerByReference destRef = new PointerByReference();

        int numOutputs = JnaUtils.imperativeInvoke(handle, src, destRef, params);
        MxNDArray[] result = new MxNDArray[numOutputs];
        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOutputs);
        for (int i = 0; i < numOutputs; i++) {
            result[i] = factory.create(ptrArray[i]);
        }
        return result;
    }

    public String getFunctionName() {
        return name;
    }

    public List<String> getArgumentNames() {
        return arguments.keys();
    }

    public List<String> getArgumentTypes() {
        return arguments.values();
    }
}
