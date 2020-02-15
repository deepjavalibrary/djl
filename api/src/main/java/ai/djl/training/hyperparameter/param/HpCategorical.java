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

import ai.djl.util.RandomUtils;
import java.util.List;

/**
 * A {@link Hyperparameter} which is one of a fixed number of options (similar to an enum).
 *
 * @param <T> the type of the options
 */
public class HpCategorical<T> extends Hyperparameter<T> {

    private List<T> categories;

    /**
     * Constructs a {@link HpCategorical}.
     *
     * @param name the name of the hyperparameters
     * @param categories the valid values for the hyperparameter
     */
    public HpCategorical(String name, List<T> categories) {
        super(name);
        this.categories = categories;
    }

    /** {@inheritDoc} */
    @Override
    public T random() {
        int index = RandomUtils.nextInt(categories.size());
        return categories.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "HPCategorical{" + "categories=" + categories + ", name='" + name + '\'' + '}';
    }
}
