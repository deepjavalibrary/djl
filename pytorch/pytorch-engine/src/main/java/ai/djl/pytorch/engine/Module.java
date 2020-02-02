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
package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.pytorch.jni.Pointer;
import ai.djl.pytorch.jni.PyTorchLibrary;

import java.nio.file.Path;
// TODO: Memory handling
public class Module {

    private Pointer handle;

    public Module(Pointer moduleHandle) {
        this.handle = moduleHandle;
    }

    public static Module load(Path path) {
        Pointer handle = PyTorchLibrary.LIB.moduleLoad(path.toString());
        return new Module(handle);
    }

    public void eval() {
        PyTorchLibrary.LIB.moduleEval(handle);
    }

    public NDList forward(NDList input) {
        Pointer[] iValueHandles = input.stream().map(ele ->
            PyTorchLibrary.LIB.iValueCreateFromTensor(((PtNDArray) ele).getHandle())
        ).toArray(Pointer[]::new);
        NDArray result = new PtNDArray(PyTorchLibrary.LIB.iValueToTensor(PyTorchLibrary.LIB.moduleForward(handle, iValueHandles)));
        return new NDList(result);
    }
}
