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
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * {@code Classifications} is the container that stores the classification results for
 * classification on a single input.
 */
public class Classifications implements JsonSerializable {

    private static final long serialVersionUID = 1L;

    private static final Gson GSON =
            JsonUtils.builder()
                    .registerTypeAdapter(Classifications.class, new ClassificationsSerializer())
                    .create();

    protected List<String> classNames;
    protected List<Double> probabilities;

    /**
     * Constructs a {@code Classifications} using a parallel list of classNames and probabilities.
     *
     * @param classNames the names of the classes
     * @param probabilities the probabilities for each class for the input
     */
    public Classifications(List<String> classNames, List<Double> probabilities) {
        this.classNames = classNames;
        this.probabilities = probabilities;
    }

    /**
     * Constructs a {@code Classifications} using list of classNames parallel to an NDArray of
     * probabilities.
     *
     * @param classNames the names of the classes
     * @param probabilities the probabilities for each class for the input
     */
    public Classifications(List<String> classNames, NDArray probabilities) {
        this.classNames = classNames;
        NDArray array = probabilities.toType(DataType.FLOAT64, false);
        this.probabilities =
                Arrays.stream(array.toDoubleArray()).boxed().collect(Collectors.toList());
        array.close();
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
    public String toJson() {
        return GSON.toJson(this) + '\n';
    }

    /** {@inheritDoc} */
    @Override
    public String getAsString() {
        return toJson();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(toJson().getBytes(StandardCharsets.UTF_8));
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('[').append(System.lineSeparator());
        for (Classification item : topK(5)) {
            sb.append('\t').append(item).append(System.lineSeparator());
        }
        sb.append(']');
        return sb.toString();
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
            if (probability < 0.00001) {
                return String.format("class: \"%s\", probability: %.1e", className, probability);
            }
            probability = (int) (probability * 100000) / 100000f;
            return String.format("class: \"%s\", probability: %.5f", className, probability);
        }
    }

    /** A customized Gson serializer to serialize the {@code Classifications} object. */
    public static final class ClassificationsSerializer implements JsonSerializer<Classifications> {

        /** {@inheritDoc} */
        @Override
        public JsonElement serialize(Classifications src, Type type, JsonSerializationContext ctx) {
            List<?> list = src.topK(5);
            return ctx.serialize(list);
        }
    }
}
