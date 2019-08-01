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

import java.util.Collections;
import java.util.List;
import java.util.Map;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

/** An interface defining neural-network layers. */
public interface Block {

    NDList forward(NDList inputs, PairList<String, String> params);

    default NDList forward(NDList inputs) {
        return forward(inputs, new PairList<>());
    }

    void backward();

    boolean isInitialized();

    Shape getInputShape();

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
            for (Block child : getChildren().values()) {
                child.ensureInitialized(inputs);
            }
        }
    }

    void beforeInitialize(NDList inputs);

    Shape getParameterShape(String name, NDList inputs);

    byte[] getEncoded();

    default Map<String, Block> getChildren() {
        return Collections.emptyMap();
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
        getChildren()
                .forEach(
                        (childName, child) -> {
                            child.getParameters()
                                    .forEach(
                                            pair -> {
                                                parameters.add(
                                                        childName + "_" + pair.getKey(),
                                                        pair.getValue());
                                            });
                        });
        return parameters;
    }
}
