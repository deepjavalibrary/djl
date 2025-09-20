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

import ai.djl.ndarray.NDArray;

/**
 * Constructs a {@link ScaledParamPatch} to scale a {@link ParamPatch} by a scalar multiplier.
 *
 * @see ParamPatch#scale(float)
 */
public class ScaledParamPatch extends ParamPatch {

    float scale;
    ParamPatch base;

    /**
     * Constructs a {@link ScaledParamPatch}.
     *
     * @param scale the scalar multiplier
     * @param base the {@link ParamPatch} to scale
     */
    public ScaledParamPatch(float scale, ParamPatch base) {
        if (base instanceof ScaledParamPatch) {
            ScaledParamPatch sbase = (ScaledParamPatch) base;
            this.scale = scale * sbase.scale;
            this.base = sbase.base;
        } else {
            this.scale = scale;
            this.base = base;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getPatch(String paramName) {
        return base.getPatch(paramName).muli(scale);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        base.close();
    }
}
