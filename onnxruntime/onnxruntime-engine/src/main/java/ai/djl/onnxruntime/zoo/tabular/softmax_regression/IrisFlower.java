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
package ai.djl.onnxruntime.zoo.tabular.softmax_regression;

/** A class holds the iris flower features. */
public class IrisFlower {

    private float sepalLength;
    private float sepalWidth;
    private float petalLength;
    private float petalWidth;

    /**
     * Constructs a new {@code IrisFlower} instance.
     *
     * @param sepalLength the sepal length
     * @param sepalWidth the sepal width
     * @param petalLength the petal length
     * @param petalWidth the petal width
     */
    public IrisFlower(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
    }

    /**
     * Returns the sepal length.
     *
     * @return the sepal length
     */
    public float getSepalLength() {
        return sepalLength;
    }

    /**
     * Returns the sepal width.
     *
     * @return the sepal width
     */
    public float getSepalWidth() {
        return sepalWidth;
    }

    /**
     * Returns the petal length.
     *
     * @return the petal length
     */
    public float getPetalLength() {
        return petalLength;
    }

    /**
     * Returns the petal width.
     *
     * @return the petal width
     */
    public float getPetalWidth() {
        return petalWidth;
    }
}
