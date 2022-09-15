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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.distribution.Distribution;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.util.PairList;

public class StudentTOutput extends DistributionOutput {

    public StudentTOutput() {
        argsDim = new PairList<>(3);
        argsDim.add("mu", 1);
        argsDim.add("sigma", 1);
        argsDim.add("nu", 1);
    }

    /** {@inheritDoc} */
    @Override
    public NDList domainMap(NDList arrays) {
        NDArray mu = arrays.get(0);
        NDArray sigma = arrays.get(1);
        NDArray nu = arrays.get(2);
        mu = mu.squeeze(-1);
        sigma = sigma.getNDArrayInternal().softPlus().squeeze(-1);
        nu = nu.getNDArrayInternal().softPlus().add(2.).squeeze(-1);
        // TODO: make setName() must be implemented
        mu.setName("mu");
        sigma.setName("sigma");
        nu.setName("nu");
        return new NDList(mu, sigma, nu);
    }

    /** {@inheritDoc} */
    @Override
    public Distribution.DistributionBuilder<?> distributionBuilder() {
        return null;
    }
}
