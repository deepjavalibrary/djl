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

    public void invoke(Pointer src, PointerByReference destRef, PairList<String, String> params) {
        JnaUtils.imperativeInvoke(handle, src, destRef, params);
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
