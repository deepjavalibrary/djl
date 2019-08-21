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
import java.util.Optional;
import software.amazon.ai.BlockList;
import software.amazon.ai.ParameterList;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
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
    public void backward() {}

    @Override
    public void setInitializer(NDManager manager, Initializer initializer) {
        setInitializer(manager, initializer, false);
    }

    @Override
    public void setInitializer(NDManager manager, Initializer initializer, boolean overwrite) {
        for (Parameter parameter : getDirectParameters()) {
            parameter.setInitializer(manager, initializer, overwrite);
        }
        for (Block child : getChildren().values()) {
            child.setInitializer(manager, initializer, overwrite);
        }
    }

    @Override
    public void setInitializer(NDManager manager, Initializer initializer, String paramName) {
        setInitializer(manager, initializer, paramName, false);
    }

    @Override
    public void setInitializer(
            NDManager manager, Initializer initializer, String paramName, boolean overwrite) {
        Optional<Parameter> parameter =
                getDirectParameters()
                        .stream()
                        .filter(pair -> pair.getName().equals(paramName))
                        .findFirst();
        if (parameter.isPresent()) {
            parameter.get().setInitializer(manager, initializer, overwrite);
        } else {
            throw new IllegalArgumentException("Could not find parameter " + paramName);
        }
    }

    @Override
    public BlockList getChildren() {
        return new BlockList();
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

    @Override
    public ParameterList getChildrenParameters() {
        ParameterList parameters = new ParameterList();
        for (Pair<String, Block> childPair : getChildren()) {
            for (Pair<String, Parameter> paramPair : childPair.getValue().getParameters()) {
                parameters.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
            }
        }
        return parameters;
    }

    protected abstract void beforeInitialize(NDList inputs);

    @Override
    public void ensureInitialized(NDList inputs) {
        if (!initialized) {
            beforeInitialize(inputs);
            for (Parameter parameter : getDirectParameters()) {
                parameter.initialize(inputs);
            }
        }
    }

    @Override
    public Block cast(DataType dataType) {
        throw new UnsupportedOperationException("Unimplemented method cast");
    }
}
