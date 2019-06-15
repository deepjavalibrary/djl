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
package com.amazon.ai.inference;

/** The <code>Classification</code> is the container to store the classification result. */
public class Classification {

    private String className;
    private double probability;

    public Classification(String className, double probability) {
        this.className = className;
        this.probability = probability;
    }

    /**
     * Return the class name.
     *
     * @return Class name
     */
    public String getClassName() {
        return className;
    }

    /**
     * Return the probability.
     *
     * <p>Probability explain how accuracy the classifier identified to the target class
     *
     * @return Probability
     */
    public double getProbability() {
        return probability;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "class: \"" + className + "\", probability: " + probability;
    }
}
