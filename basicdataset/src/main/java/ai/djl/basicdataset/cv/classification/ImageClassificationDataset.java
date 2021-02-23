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

import ai.djl.basicdataset.cv.ImageDataset;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

/**
 * A helper to create {@link ai.djl.training.dataset.Dataset}s for {@link
 * ai.djl.Application.CV#IMAGE_CLASSIFICATION}.
 */
public abstract class ImageClassificationDataset extends ImageDataset {

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public ImageClassificationDataset(ImageDataset.BaseBuilder<?> builder) {
        super(builder);
    }

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
        NDList data = new NDList(getRecordImage(manager, index));
        NDList label = new NDList(manager.create(getClassNumber(index)));
        return new Record(data, label);
    }

    /**
     * Returns the {@link ImageClassificationTranslator} matching the format of this dataset.
     *
     * @return the {@link ImageClassificationTranslator} matching the format of this dataset
     */
    public Translator<Image, Classifications> makeTranslator() {
        Pipeline pipeline = new Pipeline();

        // Resize the image if the image size is fixed
        Optional<Integer> width = getImageWidth();
        Optional<Integer> height = getImageHeight();
        if (width.isPresent() && height.isPresent()) {
            pipeline.add(new Resize(width.get(), height.get()));
        }
        pipeline.add(new ToTensor());

        return ImageClassificationTranslator.builder()
                .optSynset(getClasses())
                .setPipeline(pipeline)
                .build();
    }

    /**
     * Returns the classes that the images in the dataset are classified into.
     *
     * @return the classes that the images in the dataset are classified into
     */
    public abstract List<String> getClasses();
}
