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

import ai.djl.modality.AbstractClassifications.Item;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

@SuppressWarnings("rawtypes")
public abstract class AbstractClassifications<I extends Item> {

    protected List<String> classNames;
    protected List<Double> probabilities;

    public AbstractClassifications(List<String> classNames, List<Double> probabilities) {
        this.classNames = classNames;
        this.probabilities = probabilities;
    }

    protected abstract I item(int index);

    public List<I> items() {
        List<I> is = new ArrayList<>(classNames.size());
        for (int i = 0; i < classNames.size(); i++) {
            is.add(item(i));
        }
        return is;
    }

    public List<I> topK(int k) {
        List<I> items = items();
        items.sort(Comparator.<I>comparingDouble(Item::getProbability).reversed());
        int count = Math.min(items.size(), k);
        return items.subList(0, count);
    }

    public I best() {
        return item(probabilities.indexOf(Collections.max(probabilities)));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('[').append(System.lineSeparator());
        for (I item : topK(5)) {
            sb.append('\t').append(item).append(System.lineSeparator());
        }
        sb.append(']');
        return sb.toString();
    }

    public class Item {

        protected int index;

        protected Item(int index) {
            this.index = index;
        }

        /**
         * Returns the class name.
         *
         * @return the class name
         */
        public String getClassName() {
            return classNames.get(index);
        }

        /**
         * Returns the probability.
         *
         * <p>Probability explains how accurately the classifier identified the target class.
         *
         * @return the probability
         */
        public double getProbability() {
            return probabilities.get(index);
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            String className = getClassName();
            double probability = getProbability();
            if (probability < 0.00001) {
                return String.format("class: \"%s\", probability: %.1e", className, probability);
            }
            probability = (int) (probability * 100000) / 100000f;
            return String.format("class: \"%s\", probability: %.5f", className, probability);
        }
    }
}
