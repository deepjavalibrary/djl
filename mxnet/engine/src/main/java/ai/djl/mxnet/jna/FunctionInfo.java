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
package ai.djl.mxnet.jna;

import ai.djl.mxnet.engine.MxNDArray;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.util.List;

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
            NDManager manager, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        PointerArray srcHandles = JnaUtils.toPointerArray(src);
        PointerByReference destRef = new PointerByReference(JnaUtils.toPointerArray(dest));
        return JnaUtils.imperativeInvoke(handle, srcHandles, destRef, params).size();
    }

    public NDArray[] invoke(NDManager manager, NDArray[] src, PairList<String, ?> params) {
        PointerArray srcHandles = JnaUtils.toPointerArray(src);
        return invoke((MxNDManager) manager, srcHandles, params);
    }

    public NDArray[] invoke(NDManager manager, NDArray src, PairList<String, ?> params) {
        PointerArray handles = JnaUtils.toPointerArray(new NDArray[] {src});
        return invoke((MxNDManager) manager, handles, params);
    }

    private NDArray[] invoke(MxNDManager manager, PointerArray src, PairList<String, ?> params) {
        PointerByReference destRef = new PointerByReference();

        PairList<Pointer, SparseFormat> pairList =
                JnaUtils.imperativeInvoke(handle, src, destRef, params);
        return pairList.stream()
                .map(
                        pair -> {
                            if (pair.getValue() != SparseFormat.DENSE) {
                                return manager.create(pair.getKey(), pair.getValue());
                            }
                            return manager.create(pair.getKey());
                        })
                .toArray(MxNDArray[]::new);
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
