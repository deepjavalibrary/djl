/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.easy.cv;

import ai.djl.easy.DatasetFactory;
import java.util.List;

/** A {@link DatasetFactory} for {@link ImageClassification}. */
public interface ImageClassificationDatasetFactory extends DatasetFactory {

    /**
     * Returns the number of channels in the images in the dataset.
     *
     * <p>For example, RGB would be 3 channels while grayscale only uses 1 channel.
     *
     * @return the number of channels in the images in the dataset
     */
    int getImageChannels();

    /**
     * Returns the width of the images in the dataset.
     *
     * @return the width of the images in the dataset
     */
    int getImageWidth();

    /**
     * Returns the height of the images in the dataset.
     *
     * @return the height of the images in the dataset
     */
    int getImageHeight();

    /**
     * Returns the classes that the images in the dataset are classified into.
     *
     * @return the classes that the images in the dataset are classified into
     */
    List<String> getClasses();
}
