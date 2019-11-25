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
package ai.djl.nn;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * {@code AbstractBlock} is an abstract implementation of {@link Block}. It is recommended that all
 * {@code Block} classes that have children extend the {@code AbstractBlock}.
 */
public abstract class AbstractBlock implements Block {

    protected Shape[] inputShapes;
    protected List<String> inputNames = Collections.singletonList("data");

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (!isInitialized()) {
            throw new IllegalStateException("Parameter of this block are not initialised");
        }
        return new PairList<>(inputNames, Arrays.asList(inputShapes));
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        for (Parameter parameter : getDirectParameters()) {
            parameter.setInitializer(initializer, false);
        }
        for (Block child : getChildren().values()) {
            child.setInitializer(initializer);
        }
    }

    /** {@inheritDoc} */
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

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        ParameterList parameters = new ParameterList();
        List<Parameter> directParams = getDirectParameters();
        directParams.forEach(param -> parameters.add(param.getName(), param));
        PairList<String, Parameter> childrenParameters = getChildrenParameters();
        childrenParameters.forEach(parameters::add);
        return parameters;
    }

    /**
     * Performs any action necessary before initialization.
     *
     * @param inputShapes the expected shapes of the input
     */
    protected void beforeInitialize(Shape[] inputShapes) {
        this.inputShapes = inputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        for (Parameter param : getParameters().values()) {
            if (!param.isInitialized()) {
                return false;
            }
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        getParameters().forEach(param -> param.getValue().close());
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
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
