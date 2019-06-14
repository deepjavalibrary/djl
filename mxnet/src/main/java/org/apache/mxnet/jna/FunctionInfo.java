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

import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.util.List;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;

public class FunctionInfo {

    private Pointer handle;
    private String name;
    private PairList<String, String> arguments;

    FunctionInfo(Pointer pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    public MxNDArray[] invoke(MxNDFactory factory, PairList<String, String> params) {
        return invoke(factory, new MxNDArray[] {}, params);
    }

    public MxNDArray[] invoke(MxNDFactory factory, MxNDArray src, PairList<String, String> params) {
        return invoke(factory, new MxNDArray[] {src}, params);
    }

    public MxNDArray[] invoke(
            MxNDFactory factory, MxNDArray[] src, PairList<String, String> params) {

        Pointer[] srcHandles = new Pointer[src.length];
        for (int i = 0; i < src.length; i++) {
            srcHandles[i] = (src[i]).getHandle();
        }
        PointerArray srcRef = new PointerArray(srcHandles);

        PointerByReference destRef = new PointerByReference();

        int numOutputs = JnaUtils.imperativeInvoke(handle, srcRef, destRef, params);

        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOutputs);
        MxNDArray[] result = new MxNDArray[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            result[i] = factory.create(ptrArray[i]);
        }
        return result;
    }

    public void invoke(
            MxNDFactory factory,
            MxNDArray[] src,
            MxNDArray[] dest,
            PairList<String, String> params) {
        Pointer[] srcHandles = new Pointer[src.length];
        for (int i = 0; i < src.length; i++) {
            srcHandles[i] = (src[i]).getHandle();
        }
        PointerArray srcRef = new PointerArray(srcHandles);

        PointerByReference destRef;
        if (dest != null) {
            Pointer[] arrays = new Pointer[dest.length];
            for (int i = 0; i < dest.length; i++) {
                arrays[i] = (dest[i]).getHandle();
            }
            destRef = new PointerByReference(new PointerArray(arrays));
        } else {
            throw new NullPointerException(
                    "Please use the invoke method with a return"
                            + "type instead of passing null as destination NDArray.");
        }

        int numOutputs = JnaUtils.imperativeInvoke(handle, srcRef, destRef, params);
        if (numOutputs != dest.length) {
            throw new IllegalArgumentException(
                    "Operator output size does not match that of" + "the destination NDArray.");
        }
        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOutputs);
        for (int i = 0; i < numOutputs; i++) {
            dest[i] = factory.create(ptrArray[i]);
        }
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
