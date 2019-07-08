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
package org.apache.mxnet.nn;

import software.amazon.ai.Block;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.PairList;

public abstract class MxNNBlock implements Block {

    protected String opName;

    @Override
    public NDList forward(NDList inputs, PairList<String, String> params) {
        NDArray[] inputArray = inputs.toArray();
        NDFactory factory = inputArray[0].getFactory();
        NDArray[] output = factory.invoke(opName, inputArray, params);
        return new NDList(output);
    }

    @Override
    public void backward() {}

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
