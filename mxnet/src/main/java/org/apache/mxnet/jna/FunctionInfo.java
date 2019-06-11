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
        return invoke(factory, src, new MxNDArray[] {}, params);
    }

    public MxNDArray[] invoke(
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
        if (dest.length > 0) {
            Pointer[] arrays = new Pointer[dest.length];
            for (int i = 0; i < dest.length; i++) {
                arrays[i] = (dest[i]).getHandle();
            }
            destRef = new PointerByReference(new PointerArray(arrays));
        } else {
            destRef = new PointerByReference();
        }

        int numOutputs = JnaUtils.imperativeInvoke(handle, srcRef, destRef, params);
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
