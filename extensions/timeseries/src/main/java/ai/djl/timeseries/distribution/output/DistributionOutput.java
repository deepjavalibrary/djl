/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.distribution.output;

import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.Distribution;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.util.PairList;

public abstract class DistributionOutput {

    protected PairList<String, Integer> argsDim;
    private float valueInSupport = 0f;

    public float getValueInSupport() {
        return valueInSupport;
    }

    /**
     * Return the corresponding projection block based on the args dimension of different
     * ditributions.
     *
     * @return the corresponding projection block
     */
    public ArgProj getArgsProj() {
        return ArgProj.builder().setArgsDim(argsDim).setDomainMap(this::domainMap).build();
    }

    /**
     * Return the corresponding projection block based on the args dimension of different
     * ditributions.
     *
     * @param prefix the prefix string of projection layer block
     * @return the corresponding projection block
     */
    public ArgProj getArgsProj(String prefix) {
        return ArgProj.builder()
                .setArgsDim(argsDim)
                .setDomainMap(this::domainMap)
                .optPrefix(prefix)
                .build();
    }

    /**
     * Convert arguments to the right shape and domain. The domain depends on the type of
     * distribution, while the correct shape is obtained by reshaping the trailing axis in such a
     * way that the returned tensors define a distribution of the right event_shape.
     *
     * <p>This function is usually used as the lambda of the Lambda block.
     *
     * @param arrays the arguments
     * @return converted arguments
     */
    public abstract NDList domainMap(NDList arrays);

    /**
     * Return the associated {@code DistributionBuilder}, given the collection of constructor arguments and, optionally, a scale tensor.
     *
     * @return the associated {@code DistributionBuilder}
     */
    public abstract Distribution.DistributionBuilder<?> distributionBuilder();
}
