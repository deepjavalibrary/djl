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
package com.amazon.ai;

import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDFuncParams;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.util.PairList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public interface Block {

    NDList forward(NDList inputs, PairList<String, String> params, NDFuncParams fparams);

    void backward();

    Shape getInputShape();

    List<NDArray> getDirectParameters();

    void initialize(NDFactory factory, Initializer initializer);

    byte[] getEncoded();

    default NDList forward(NDList inputs) {
        return forward(inputs, new PairList<>(), NDFuncParams.NONE);
    }

    default NDList forward(NDList inputs, PairList<String, String> params) {
        return forward(inputs, params, NDFuncParams.NONE);
    }

    default List<Block> getChildren() {
        return Collections.emptyList();
    }

    default List<NDArray> getParameters() {
        List<NDArray> parameters = new ArrayList<>();
        parameters.addAll(getChildrenParameters());
        parameters.addAll(getDirectParameters());
        return parameters;
    }

    default List<NDArray> getChildrenParameters() {
        List<NDArray> parameters = new ArrayList<>();
        for (Block child : getChildren()) {
            parameters.addAll(child.getParameters());
        }
        return parameters;
    }
}
