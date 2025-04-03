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
import ai.djl.translate.Ensembleable;
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonElement;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * {@code Classifications} is the container that stores the classification results for
 * classification on a single input.
 */
public class Classifications implements JsonSerializable, Ensembleable<Classifications> {

    private static final long serialVersionUID = 1L;

    @SuppressWarnings("serial")
    protected List<String> classNames;

    @SuppressWarnings("serial")
    protected List<Double> probabilities;

    protected int topK;

    /**
     * Constructs a {@code Classifications} using a parallel list of classNames and probabilities.
     *
     * @param classNames the names of the classes
     * @param probabilities the probabilities for each class for the input
     */
    public Classifications(List<String> classNames, List<Double> probabilities) {
        this.classNames = classNames;
        this.probabilities = probabilities;
        this.topK = 5;
    }

    /**
     * Constructs a {@code Classifications} using list of classNames parallel to an NDArray of
     * probabilities.
     *
     * @param classNames the names of the classes
     * @param probabilities the probabilities for each class for the input
     */
    public Classifications(List<String> classNames, NDArray probabilities) {
        this(classNames, probabilities, 5);
    }

    /**
     * Constructs a {@code Classifications} using list of classNames parallel to an NDArray of
     * probabilities.
     *
     * @param classNames the names of the classes
     * @param probabilities the probabilities for each class for the input
     * @param topK the number of top classes to return
     */
    public Classifications(List<String> classNames, NDArray probabilities, int topK) {
        this.classNames = classNames;
        if (probabilities.getDataType() == DataType.FLOAT32) {
            // Avoid converting float32 to float64 as this is not supported on MPS device
            this.probabilities = new ArrayList<>();
            for (float prob : probabilities.toFloatArray()) {
                this.probabilities.add((double) prob);
            }
        } else {
            NDArray array = probabilities.toType(DataType.FLOAT64, false);
            this.probabilities =
                    Arrays.stream(array.toDoubleArray()).boxed().collect(Collectors.toList());
            array.close();
        }
        this.topK = topK;
    }

    /**
     * Returns the classes that were classified into.
     *
     * @return the classes that were classified into
     */
    public List<String> getClassNames() {
        return classNames;
    }

    /**
     * Returns the list of probabilities for each class (matching the order of the class names).
     *
     * @return the list of probabilities for each class (matching the order of the class names)
     */
    public List<Double> getProbabilities() {
        return probabilities;
    }

    /**
     * Set the topK number of classes to be displayed.
     *
     * @param topK the number of top classes to return
     */
    public final void setTopK(int topK) {
        this.topK = topK;
    }

    /**
     * Returns a classification item for each potential class for the input.
     *
     * @param <T> the type of classification item for the task
     * @return the list of classification items
     */
    public <T extends Classification> List<T> items() {
        List<T> list = new ArrayList<>(classNames.size());
        for (int i = 0; i < classNames.size(); i++) {
            list.add(item(i));
        }
        return list;
    }

    /**
     * Returns the item at a given index based on the order used to construct the {@link
     * Classifications}.
     *
     * @param index the index of the item to return
     * @param <T> the type of classification item for the task
     * @return the item at the given index, equivalent to {@code classifications.items().get(index)}
     */
    @SuppressWarnings("unchecked")
    public <T extends Classification> T item(int index) {
        return (T) new Classification(classNames.get(index), probabilities.get(index));
    }

    /**
     * Returns a list of the top classes.
     *
     * @param <T> the type of the classification item for the task
     * @return the list of classification items for the best classes in order of best to worst
     */
    public <T extends Classification> List<T> topK() {
        return topK(topK);
    }

    /**
     * Returns a list of the top {@code k} best classes.
     *
     * @param k the number of classes to return
     * @param <T> the type of the classification item for the task
     * @return the list of classification items for the best classes in order of best to worst
     */
    public <T extends Classification> List<T> topK(int k) {
        List<T> items = items();
        items.sort(Comparator.comparingDouble(Classification::getProbability).reversed());
        int count = Math.min(items.size(), k);
        return items.subList(0, count);
    }

    /**
     * Returns the most likely class for the classification.
     *
     * @param <T> the type of the classification item for the task
     * @return the classification item
     */
    public <T extends Classification> T best() {
        return item(probabilities.indexOf(Collections.max(probabilities)));
    }

    /**
     * Returns the result for a particular class name.
     *
     * @param className the class name to get results for
     * @param <T> the type of the classification item for the task
     * @return the (first if multiple) classification item
     */
    public <T extends Classification> T get(String className) {
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
    public JsonElement serialize() {
        return JsonUtils.GSON.toJsonTree(topK());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        List<Classification> list = topK();
        int index = 0;
        for (Classification item : list) {
            sb.append('\t').append(item);
            if (++index < list.size()) {
                sb.append(',');
            }
            sb.append('\n');
        }
        sb.append("]\n");
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public Classifications ensembleWith(Iterator<Classifications> it) {
        int size = probabilities.size();
        List<Double> newProbabilities = new ArrayList<>(size);
        newProbabilities.addAll(probabilities);
        int count = 1;
        while (it.hasNext()) {
            ++count;
            Classifications c = it.next();
            for (int i = 0; i < size; ++i) {
                newProbabilities.set(i, newProbabilities.get(i) + c.probabilities.get(i));
            }
            if (!c.classNames.equals(classNames)) {
                throw new IllegalArgumentException(
                        "Found a classNames mismatch during ensembling. All input Classifications"
                                + " should have the same classNames, but some were different");
            }
        }
        final int total = count;
        newProbabilities.replaceAll(p -> p / total);
        return new Classifications(classNames, newProbabilities);
    }

    /**
     * A {@code Classification} stores the classification result for a single class on a single
     * input.
     */
    public static class Classification {

        private String className;
        private double probability;

        /**
         * Constructs a single class result for a classification.
         *
         * @param className the class name of the result
         * @param probability the probability of the result
         */
        public Classification(String className, double probability) {
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
            StringBuilder sb = new StringBuilder(100);
            sb.append("{\"className\": \"").append(className).append("\", \"probability\": ");
            if (probability < 0.00001) {
                sb.append(String.format("%.1e", probability));
            } else {
                probability = (int) (probability * 100000) / 100000f;
                sb.append(String.format("%.5f", probability));
            }
            sb.append('}');
            return sb.toString();
        }
    }
}
