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

/**
 * A {@link Hyperparameter} with a known value instead of a range of possible values.
 *
 * <p>{@link HpVal}s and {@link HpSet}s of {@link HpVal}s are used to represent sampled
 * hyperparameters.
 *
 * @param <T> the type of the value
 */
public class HpVal<T> extends Hyperparameter<T> {

    T value;

    /**
     * Cosntructs a new {@link HpVal}.
     *
     * @param name the name of the hyperparameter
     * @param value the fixed value of the hyperparameter
     */
    public HpVal(String name, T value) {
        super(name);
        this.value = value;
    }

    /** {@inheritDoc} */
    @Override
    public T random() {
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "HPVal{" + "value=" + value + ", name='" + name + '\'' + '}';
    }
}
