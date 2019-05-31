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
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.Shape;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface Block {

    default NDList forward(NDList inputs) {
        return forward(inputs, Collections.emptyMap());
    }

    NDList forward(NDList inputs, Map<String, String> args);

    void backward();

    Shape getInputShape();

    List<NDArray> getParameters();

    byte[] getEncoded();
}
