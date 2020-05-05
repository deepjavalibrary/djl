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

import ai.djl.ndarray.NDArray;
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
    int getHeigth();

    /**
     * Converts image to {@link NDArray}.
     *
     * @return {@link NDArray}
     */
    NDArray toNDArray();

    /**
     * Save the image to file.
     *
     * @param os {@link OutputStream} to save the image.
     * @param type type of the image, such as "png", "jpeg"
     */
    void save(OutputStream os, String type);
}
