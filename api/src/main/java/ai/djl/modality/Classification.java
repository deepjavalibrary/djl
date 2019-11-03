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
package ai.djl.modality;

import ai.djl.modality.Classification.Item;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/** {@code Classification} is the container that stores the classification results. */
public final class Classification extends AbstractClassifications<Item> {

    public Classification(List<String> classNames, List<Double> probabilities) {
        super(classNames, probabilities);
    }

    public Classification(List<String> classNames, NDArray probabilities) {
        super(
                classNames,
                Arrays.stream(probabilities.asType(DataType.FLOAT64, false).toDoubleArray())
                        .boxed()
                        .collect(Collectors.toList()));
    }

    public Item get(String className) {
        for (int i = 0; i < classNames.size(); i++) {
            if (classNames.get(i).equals(className)) {
                return item(i);
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    protected Item item(int index) {
        return new Item(index);
    }

    public final class Item extends AbstractClassifications<Item>.Item {

        protected Item(int index) {
            super(index);
        }
    }
}
