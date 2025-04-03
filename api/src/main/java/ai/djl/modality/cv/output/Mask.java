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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

/**
 * A mask with a probability for each pixel within a bounding rectangle.
 *
 * <p>This class is usually used to record the results of an Image Segmentation task.
 */
public class Mask extends Rectangle {

    private static final long serialVersionUID = 1L;
    private float[][] probDist;
    private boolean fullImageMask;

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
        this(x, y, width, height, dist, false);
    }

    /**
     * Constructs a Mask with the given data.
     *
     * @param x the left coordinate of the bounding rectangle
     * @param y the top coordinate of the bounding rectangle
     * @param width the width of the bounding rectangle
     * @param height the height of the bounding rectangle
     * @param dist the probability distribution for each pixel in the rectangle
     * @param fullImageMask if the mask if for full image
     */
    public Mask(
            double x,
            double y,
            double width,
            double height,
            float[][] dist,
            boolean fullImageMask) {
        super(x, y, width, height);
        this.probDist = dist;
        this.fullImageMask = fullImageMask;
    }

    /**
     * Returns the probability for each pixel.
     *
     * @return the probability for each pixel
     */
    public float[][] getProbDist() {
        return probDist;
    }

    /**
     * Returns if the mask is for full image.
     *
     * @return if the mask is for full image
     */
    public boolean isFullImageMask() {
        return fullImageMask;
    }

    /** {@inheritDoc} */
    @Override
    public JsonObject serialize() {
        JsonObject ret = super.serialize();
        if (fullImageMask) {
            ret.add("fullImageMask", new JsonPrimitive(true));
        }
        ret.add("mask", JsonUtils.GSON.toJsonTree(probDist));
        return ret;
    }

    /**
     * Converts the mask tensor to a mask array.
     *
     * @param array the mask NDArray
     * @return the mask array
     */
    public static float[][] toMask(NDArray array) {
        Shape maskShape = array.getShape();
        int height = (int) maskShape.get(0);
        int width = (int) maskShape.get(1);
        float[] flattened = array.toFloatArray();
        float[][] mask = new float[height][width];
        for (int i = 0; i < height; i++) {
            System.arraycopy(flattened, i * width, mask[i], 0, width);
        }
        return mask;
    }
}
