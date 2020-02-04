/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.hyperparameter.optimizer;

import ai.djl.training.hyperparameter.param.HpSet;
import ai.djl.util.Pair;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A base containing shared implementations for {@link HpOptimizer}s.
 *
 * @see HpOptimizer
 */
public abstract class BaseHpOptimizer implements HpOptimizer {

    protected HpSet hyperParams;
    protected Map<HpSet, Float> results;

    /**
     * Constructs a {@link BaseHpOptimizer}.
     *
     * @param hyperParams the set of hyperparameters
     */
    public BaseHpOptimizer(HpSet hyperParams) {
        this.hyperParams = hyperParams;
        results = new LinkedHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(HpSet config, float loss) {
        results.compute(config, (k, oldLoss) -> oldLoss != null ? Math.max(oldLoss, loss) : loss);
    }

    /** {@inheritDoc} */
    @Override
    public float getLoss(HpSet config) {
        return results.get(config);
    }

    /** {@inheritDoc} */
    @Override
    public Pair<HpSet, Float> getBest() {
        Entry<HpSet, Float> entry =
                Collections.min(
                        results.entrySet(),
                        (e1, e2) -> Float.compare(e1.getValue(), e2.getValue()));
        return new Pair<>(entry.getKey(), entry.getValue());
    }
}
