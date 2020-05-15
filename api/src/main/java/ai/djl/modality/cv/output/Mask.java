/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.output;

/**
 * A mask with a probability for each pixel within a bounding rectangle.
 *
 * <p>This class is usually used to record the results of an Image Segmentation task.
 */
public class Mask extends Rectangle {

    private static final long serialVersionUID = 1L;
    private float[][] probDist;

    /**
     * Constructs a Mask with the given data.
     *
     * @param x the left coordinate of the bounding rectangle
     * @param y the top coordinate of the bounding rectangle
     * @param width the width of the bounding rectangle
     * @param height the height of the bounding rectangle
     * @param dist the probability distribution for each pixel in the rectangle
     */
    public Mask(double x, double y, double width, double height, float[][] dist) {
        super(x, y, width, height);
        this.probDist = dist;
    }

    /**
     * Returns the probability for each pixel.
     *
     * @return the probability for each pixel
     */
    public float[][] getProbDist() {
        return probDist;
    }
}
