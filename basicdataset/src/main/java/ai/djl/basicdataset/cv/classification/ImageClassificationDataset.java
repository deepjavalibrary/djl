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
package ai.djl.basicdataset.cv.classification;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import java.io.IOException;

/**
 * A helper to create {@link ai.djl.training.dataset.Dataset}s for {@link
 * ai.djl.Application.CV#IMAGE_CLASSIFICATION}.
 */
public abstract class ImageClassificationDataset extends RandomAccessDataset {

    Image.Flag flag;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public ImageClassificationDataset(BaseBuilder<?> builder) {
        super(builder);
        this.flag = builder.flag;
    }

    /**
     * Returns the image at the given index in the dataset.
     *
     * @param index the index (if the dataset is a list of data items)
     * @return the image
     * @throws IOException if the image could not be loaded
     */
    protected abstract Image getImage(long index) throws IOException;

    /**
     * Returns the class of the data item at the given index.
     *
     * @param index the index (if the dataset is a list of data items)
     * @return the class number or the index into the list of classes of the desired class name
     * @throws IOException if the data could not be loaded
     */
    protected abstract long getClassNumber(long index) throws IOException;

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDList data = new NDList(getImage(index).toNDArray(manager, flag));
        NDList label = new NDList(manager.create(getClassNumber(index)));
        return new Record(data, label);
    }

    /**
     * Used to build an {@link ImageClassificationDataset}.
     *
     * @param <T> the builder type
     */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder<T>>
            extends RandomAccessDataset.BaseBuilder<T> {

        Image.Flag flag;

        protected BaseBuilder() {
            flag = Image.Flag.COLOR;
        }

        /**
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public T optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
        }
    }
}
