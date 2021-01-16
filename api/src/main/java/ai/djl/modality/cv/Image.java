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
package ai.djl.modality.cv;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.io.IOException;
import java.io.OutputStream;

/**
 * {@code Image} is a container of an image in DJL. The storage type of the image depends on the
 * platform you are running on.
 */
public interface Image {

    /**
     * Gets the width of the image.
     *
     * @return pixels representing width
     */
    int getWidth();

    /**
     * Gets the height of the image.
     *
     * @return pixels representing height
     */
    int getHeight();

    /**
     * Gets the wrapped image.
     *
     * @return the wrapped image object
     */
    Object getWrappedImage();

    /**
     * Gets the subimage defined by a specified rectangular region.
     *
     * @param x the X coordinate of the upper-left corner of the specified rectangular region
     * @param y the Y coordinate of the upper-left corner of the specified rectangular region
     * @param w the width of the specified rectangular region
     * @param h the height of the specified rectangular region
     * @return subimage of this image
     */
    Image getSubimage(int x, int y, int w, int h);

    /**
     * Gets a deep copy of the original image given the type.
     *
     * @param type the type of the copied image
     * @return the copy of the original image.
     */
    Image duplicate(Type type);

    /**
     * Converts image to a RGB {@link NDArray}.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @return {@link NDArray}
     */
    default NDArray toNDArray(NDManager manager) {
        return toNDArray(manager, null);
    }

    /**
     * Converts image to a {@link NDArray}.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param flag the color mode
     * @return {@link NDArray}
     */
    NDArray toNDArray(NDManager manager, Flag flag);

    /**
     * Save the image to file.
     *
     * @param os {@link OutputStream} to save the image.
     * @param type type of the image, such as "png", "jpeg"
     * @throws IOException image cannot be saved through output stream
     */
    void save(OutputStream os, String type) throws IOException;

    /**
     * Draws the bounding boxes on the image.
     *
     * @param detections the object detection results
     */
    void drawBoundingBoxes(DetectedObjects detections);

    /**
     * Draws all joints of a body on an image.
     *
     * @param joints the joints of the body
     */
    void drawJoints(Joints joints);

    /** Flag indicates the color channel options for images. */
    enum Flag {
        GRAYSCALE,
        COLOR;

        /**
         * Returns the number of channels for this flag.
         *
         * @return the number of channels for this flag
         */
        public int numChannels() {
            switch (this) {
                case GRAYSCALE:
                    return 1;
                case COLOR:
                    return 3;
                default:
                    throw new IllegalArgumentException("Invalid FLAG");
            }
        }
    }

    /** Type indicates the type options for images. */
    enum Type {
        TYPE_INT_ARGB
    }

    /** Interpolation indicates the Interpolation options for resizinig an image. */
    enum Interpolation {
        NEAREST,
        BILINEAR,
        AREA,
        BICUBIC
    }
}
