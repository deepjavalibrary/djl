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
package software.amazon.ai;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDScopedFactory;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

/** An interface defining neural-network layers. */
public interface Block {

    NDList forward(NDList inputs, PairList<String, String> params);

    void backward();

    Shape getInputShape();

    List<NDArray> getDirectParameters();

    void initialize(NDScopedFactory factory, Initializer initializer);

    byte[] getEncoded();

    default NDList forward(NDList inputs) {
        return forward(inputs, new PairList<>());
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
