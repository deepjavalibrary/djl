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

import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

/** An interface defining neural-network layers. */
public interface Block {

    NDList forward(NDList inputs, PairList<String, Object> params);

    default NDList forward(NDList inputs) {
        return forward(inputs, new PairList<>());
    }

    default void backward() {}

    boolean isInitialized();

    Shape getOutputShape(Shape... inputs);

    List<Parameter> getDirectParameters();

    default void setInitializer(NDManager manager, Initializer initializer) {
        for (Parameter parameter : getDirectParameters()) {
            parameter.setInitializer(manager, initializer);
        }
        for (Block child : getChildren().values()) {
            child.setInitializer(manager, initializer);
        }
    }

    default void ensureInitialized(NDList inputs) {
        if (!isInitialized()) {
            beforeInitialize(inputs);
            for (Parameter parameter : getDirectParameters()) {
                parameter.initialize(inputs);
            }
        }
    }

    void beforeInitialize(NDList inputs);

    default Block cast(DataType dataType) {
        throw new UnsupportedOperationException("Unimplemented method cast");
    }

    DataDesc[] describeInput();

    Shape getParameterShape(String name, NDList inputs);

    byte[] getEncoded();

    default PairList<String, Block> getChildren() {
        return new PairList<>();
    }

    default PairList<String, Parameter> getParameters() {
        PairList<String, Parameter> parameters = new PairList<>();
        List<Parameter> directParams = this.getDirectParameters();
        directParams.forEach(param -> parameters.add(param.getName(), param));
        PairList<String, Parameter> childrenParameters = getChildrenParameters();
        childrenParameters.forEach(pair -> parameters.add(pair));
        return parameters;
    }

    default PairList<String, Parameter> getChildrenParameters() {
        PairList<String, Parameter> parameters = new PairList<>();
        for (Pair<String, Block> childPair : getChildren()) {
            for (Pair<String, Parameter> paramPair : childPair.getValue().getParameters()) {
                parameters.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
            }
        }
        return parameters;
    }
}
