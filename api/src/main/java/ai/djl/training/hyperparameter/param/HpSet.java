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
package ai.djl.training.hyperparameter.param;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A nestable set of {@link Hyperparameter}s. */
public final class HpSet extends Hyperparameter<HpSet> {

    private Map<String, Hyperparameter<?>> hyperParams;

    /**
     * Cosntructs a new {@link HpSet}.
     *
     * @param name the name of the hyperparameter set
     * @param hyperParams the included hyperparameters in the set
     */
    public HpSet(String name, List<Hyperparameter<?>> hyperParams) {
        super(name);
        this.hyperParams = new ConcurrentHashMap<>();
        for (Hyperparameter<?> hparam : hyperParams) {
            add(hparam);
        }
    }

    /**
     * Cosntructs a new empty {@link HpSet}.
     *
     * @param name the name of the hyperparameter set
     */
    public HpSet(String name) {
        super(name);
        hyperParams = new ConcurrentHashMap<>();
    }

    /**
     * Adds a hyperparameter to the set.
     *
     * @param hparam the hyperparameter to add
     */
    public void add(Hyperparameter<?> hparam) {
        hyperParams.put(hparam.getName(), hparam);
    }

    /**
     * Returns the hyperparameter in the set with the given name.
     *
     * @param name the name of the hyperparameter to return
     * @return the hyperparameter
     */
    public Hyperparameter<?> getHParam(String name) {
        return hyperParams.get(name);
    }

    /** {@inheritDoc} */
    @Override
    public HpSet random() {
        HpSet rand = new HpSet(name);
        for (Hyperparameter<?> hparam : hyperParams.values()) {
            rand.add(new HpVal<>(hparam.getName(), hparam.random()));
        }
        return rand;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "HPSet{" + "hyperParams=" + hyperParams + ", name='" + name + '\'' + '}';
    }
}
