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
package ai.djl.basicdataset.cv;

import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Record;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.IOException;

/**
 * A helper to create {@link ai.djl.training.dataset.Dataset}s for {@link
 * ai.djl.Application.CV#OBJECT_DETECTION}.
 */
public abstract class ObjectDetectionDataset extends ImageDataset {

    /**
     * Creates a new instance of {@link ObjectDetectionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public ObjectDetectionDataset(ImageDataset.BaseBuilder<?> builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDList data = new NDList(getRecordImage(manager, index));

        PairList<Long, Rectangle> objects = getObjects(index);
        float[][] labelsSplit = new float[objects.size()][5];
        for (int i = 0; i < objects.size(); i++) {
            Pair<Long, Rectangle> obj = objects.get(i);
            labelsSplit[i][0] = obj.getKey();

            Rectangle location = obj.getValue();
            labelsSplit[i][1] = (float) location.getX();
            labelsSplit[i][2] = (float) location.getY();
            labelsSplit[i][3] = (float) location.getWidth();
            labelsSplit[i][4] = (float) location.getHeight();
        }
        NDList labels = new NDList(manager.create(labelsSplit));
        return new Record(data, labels);
    }

    /**
     * Returns the list of objects in the image at the given index.
     *
     * @param index the index (if the dataset is a list of data items)
     * @return the list of objects in the image. The long is the class number of the index into the
     *     list of classes of the desired class name. The rectangle is the location of the object
     *     inside the image.
     * @throws IOException if the data could not be loaded
     */
    public abstract PairList<Long, Rectangle> getObjects(long index) throws IOException;
}
