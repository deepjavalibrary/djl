/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.patch;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.util.Pair;

/**
 * A standard {@link Patch} that only adds to {@link Parameter}s.
 *
 * <p>To create a param patch, see {@link BasicParamPatch}.
 */
public abstract class ParamPatch extends ReversiblePatch {

    /**
     * Scales the patch by a scalar multiplier.
     *
     * @param scale the scalar multiplier for each patch NDArray
     * @return a new patch that is a scaled version of this patch
     */
    public ParamPatch scale(float scale) {
        return new ScaledParamPatch(scale, this);
    }

    /** {@inheritDoc} */
    @Override
    public ParamPatch reverse() {
        return scale(-1);
    }

    /**
     * Returns the patch {@link NDArray} for a particular paramName.
     *
     * @param paramName the parameter path in a {@link ParameterList}.
     * @return the patch array
     */
    public abstract NDArray getPatch(String paramName);

    /**
     * Applies the part of this patch to a particular {@link Parameter}.
     *
     * @param paramName the parameter path in a {@link ParameterList}.
     * @param param the {@link Parameter} to patch
     */
    public void apply(String paramName, Parameter param) {
        NDArray p = getPatch(paramName).duplicate();
        param.getArray().addi(p);
        p.close();
    }

    /**
     * Applies this patch to a {@link ParameterList}.
     *
     * @param params the params to patch
     */
    public void apply(ParameterList params) {
        for (Pair<String, Parameter> param : params) {
            apply(param.getKey(), param.getValue());
        }
    }

    /**
     * Applies this patch to a {@link Block}.
     *
     * @param block the block to patch
     */
    public void apply(Block block) {
        apply(block.getParameters());
    }

    /**
     * Applies this patch to a {@link Model}.
     *
     * @param model the model to patch
     */
    @Override
    public void apply(Model model) {
        apply(model.getBlock());
    }
}
