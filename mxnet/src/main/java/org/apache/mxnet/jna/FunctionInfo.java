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

import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.util.List;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;

public class FunctionInfo {

    private Pointer handle;
    private String functionName;
    private List<String> arguments;
    private List<String> signature;

    FunctionInfo(
            Pointer pointer, String functionName, List<String> arguments, List<String> signature) {
        this.handle = pointer;
        this.functionName = functionName;
        this.arguments = arguments;
        this.signature = signature;
    }

    public NDList invoke(MxNDFactory factory, MxNDArray src, PairList<String, String> params) {
        return invoke(factory, new NDList(src), null, params);
    }

    public NDList invoke(MxNDFactory factory, NDList src, PairList<String, String> params) {
        return invoke(factory, src, null, params);
    }

    public NDList invoke(
            MxNDFactory factory, MxNDArray src, NDList dest, PairList<String, String> params) {
        return invoke(factory, new NDList(src), dest, params);
    }

    public NDList invoke(
            MxNDFactory factory, NDList src, NDList dest, PairList<String, String> params) {

        Pointer[] srcArray = new Pointer[src.size()];
        for (int i = 0; i < src.size(); i++) {
            srcArray[i] = ((MxNDArray) src.get(i)).getHandle();
        }
        PointerArray srcRef = new PointerArray(srcArray);

        PointerByReference destRef;
        if (dest != null) {
            Pointer[] arrays = new Pointer[dest.size()];
            for (int i = 0; i < dest.size(); i++) {
                arrays[i] = ((MxNDArray) dest.get(i)).getHandle();
            }
            destRef = new PointerByReference(new PointerArray(arrays));
        } else {
            destRef = new PointerByReference();
        }

        int numOutputs = JnaUtils.imperativeInvoke(handle, srcRef, destRef, params);

        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOutputs);
        NDList result = new NDList();
        for (int i = 0; i < numOutputs; i++) {
            result.add(factory.create(ptrArray[i]));
        }
        return result;
    }

    public String getFunctionName() {
        return functionName;
    }

    public List<String> getArguments() {
        return arguments;
    }

    public List<String> getSignature() {
        return signature;
    }
}
