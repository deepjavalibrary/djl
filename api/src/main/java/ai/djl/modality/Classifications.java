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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/** {@code Classification} is the container that stores the classification results. */
public class Classifications {

    protected List<String> classNames;
    protected List<Double> probabilities;

    public Classifications(List<String> classNames, List<Double> probabilities) {
        this.classNames = classNames;
        this.probabilities = probabilities;
    }

    public Classifications(List<String> classNames, NDArray probabilities) {
        this.classNames = classNames;
        NDArray array = probabilities.asType(DataType.FLOAT64, false);
        this.probabilities =
                Arrays.stream(array.toDoubleArray()).boxed().collect(Collectors.toList());
        array.close();
    }

    public <T extends Item> List<T> items() {
        List<T> list = new ArrayList<>(classNames.size());
        for (int i = 0; i < classNames.size(); i++) {
            list.add(item(i));
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    public <T extends Item> T item(int index) {
        return (T) new Item(classNames.get(index), probabilities.get(index));
    }

    public <T extends Item> List<T> topK(int k) {
        List<T> items = items();
        items.sort(Comparator.comparingDouble(Item::getProbability).reversed());
        int count = Math.min(items.size(), k);
        return items.subList(0, count);
    }

    public <T extends Item> T best() {
        return item(probabilities.indexOf(Collections.max(probabilities)));
    }

    public <T extends Item> T get(String className) {
        int size = classNames.size();
        for (int i = 0; i < size; i++) {
            if (classNames.get(i).equals(className)) {
                return item(i);
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('[').append(System.lineSeparator());
        for (Item item : topK(5)) {
            sb.append('\t').append(item).append(System.lineSeparator());
        }
        sb.append(']');
        return sb.toString();
    }

    public static class Item {

        private String className;
        private double probability;

        public Item(String className, double probability) {
            this.className = className;
            this.probability = probability;
        }

        /**
         * Returns the class name.
         *
         * @return the class name
         */
        public String getClassName() {
            return className;
        }

        /**
         * Returns the probability.
         *
         * <p>Probability explains how accurately the classifier identified the target class.
         *
         * @return the probability
         */
        public double getProbability() {
            return probability;
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            if (probability < 0.00001) {
                return String.format("class: \"%s\", probability: %.1e", className, probability);
            }
            probability = (int) (probability * 100000) / 100000f;
            return String.format("class: \"%s\", probability: %.5f", className, probability);
        }
    }
}
