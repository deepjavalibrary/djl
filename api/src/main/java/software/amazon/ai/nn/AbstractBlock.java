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
package software.amazon.ai.nn;

import java.util.List;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

public abstract class AbstractBlock implements Block {

    protected boolean initialized;

    @Override
    public NDList forward(NDList inputs) {
        return forward(inputs, new PairList<>());
    }

    @Override
    public DataDesc[] describeInput() {
        return new DataDesc[0];
    }

    @Override
    public void backward() {}

    @Override
    public void setInitializer(Initializer initializer) {
        for (Parameter parameter : getDirectParameters()) {
            parameter.setInitializer(initializer, false);
        }
        for (Block child : getChildren().values()) {
            child.setInitializer(initializer);
        }
    }

    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        Parameter parameter =
                getDirectParameters()
                        .stream()
                        .filter(pair -> pair.getName().equals(paramName))
                        .findFirst()
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "Could not find parameter " + paramName));
        parameter.setInitializer(initializer, true);
    }

    @Override
    public ParameterList getParameters() {
        ParameterList parameters = new ParameterList();
        List<Parameter> directParams = getDirectParameters();
        directParams.forEach(param -> parameters.add(param.getName(), param));
        PairList<String, Parameter> childrenParameters = getChildrenParameters();
        childrenParameters.forEach(parameters::add);
        return parameters;
    }

    protected void beforeInitialize(Shape[] inputShapes) {}

    @Override
    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Unimplemented method cast");
    }

    @Override
    public void clear() {
        getParameters().forEach(param -> param.getValue().close());
    }

    private ParameterList getChildrenParameters() {
        ParameterList parameters = new ParameterList();
        for (Pair<String, Block> childPair : getChildren()) {
            for (Pair<String, Parameter> paramPair : childPair.getValue().getParameters()) {
                parameters.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
            }
        }
        return parameters;
    }
}
