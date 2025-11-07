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
import ai.djl.training.GradientCollector;
import ai.djl.util.Pair;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The basic implementation of a {@link ParamPatch}. */
public class BasicParamPatch extends ParamPatch {

    Map<String, NDArray> data;

    /**
     * Constructs a {@link BasicParamPatch} with patching data.
     *
     * @param data the patching data
     */
    public BasicParamPatch(Map<String, NDArray> data) {
        this.data = data;
    }

    /**
     * Makes a patch by comparing two models.
     *
     * @param source the source model
     * @param target the target model
     * @return a patch that would transform the source model to the target model
     */
    public static BasicParamPatch makePatch(Model source, Model target) {
        return BasicParamPatch.makePatch(source.getBlock(), target.getBlock());
    }

    /**
     * Makes a patch by comparing two blocks.
     *
     * @param source the source block
     * @param target the target block
     * @return a patch that would transform the source block to the target block
     */
    public static BasicParamPatch makePatch(Block source, Block target) {
        return BasicParamPatch.makePatch(source.getParameters(), target.getParameters());
    }

    /**
     * Makes a patch by comparing two {@link ParameterList}s.
     *
     * @param source the source {@link ParameterList}
     * @param target the target {@link ParameterList}
     * @return a patch that would transform the source {@link ParameterList} to the target {@link
     *     ParameterList}.
     */
    public static BasicParamPatch makePatch(ParameterList source, ParameterList target) {
        Map<String, NDArray> data = new ConcurrentHashMap<>(source.size());
        for (Pair<String, Parameter> sourcePair : source) {
            String key = sourcePair.getKey();
            NDArray patchValue = target.get(key).getArray().sub(sourcePair.getValue().getArray());
            data.put(key, patchValue);
        }
        return new BasicParamPatch(data);
    }

    /**
     * Makes a patch from gradients.
     *
     * <p>This does not include learning rates or any other data from the {@link
     * ai.djl.training.optimizer.Optimizer}.
     *
     * <p>Making the patch does not modify the existing gradients. After this, you can call {@link
     * GradientCollector#zeroGradients()} to clear the gradients.
     *
     * @param block the block for which to collect gradients
     * @param gradientCollector the {@link GradientCollector} of the gradients
     * @return the gradients as a {@link BasicParamPatch}.
     */
    public static BasicParamPatch makePatch(Block block, GradientCollector gradientCollector) {
        ParameterList params = block.getParameters();
        Map<String, NDArray> data = new ConcurrentHashMap<>(params.size());
        for (Pair<String, Parameter> param : params) {
            String key = param.getKey();
            // Get gradient * -1 to account for gradient being subtracted from param
            NDArray patchValue = param.getValue().getArray().getGradient().duplicate().mul(-1);
            data.put(key, patchValue);
        }
        return new BasicParamPatch(data);
    }

    /**
     * Makes a patch from gradients.
     *
     * <p>This does not include learning rates or any other data from the {@link
     * ai.djl.training.optimizer.Optimizer}.
     *
     * <p>Making the patch does not modify the existing gradients. After this, you can call {@link
     * GradientCollector#zeroGradients()} to clear the gradients.
     *
     * @param model the model for which to collect gradients
     * @param gradientCollector the {@link GradientCollector} of the gradients
     * @return the gradients as a {@link BasicParamPatch}.
     */
    public static BasicParamPatch makePatch(Model model, GradientCollector gradientCollector) {
        return makePatch(model.getBlock(), gradientCollector);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getPatch(String paramName) {
        return data.get(paramName).duplicate();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (NDArray d : data.values()) {
            d.close();
        }
    }
}
