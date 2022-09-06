/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.dataset;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * BulkDataIterable specializes DataIterable in using {@link ArrayDataset#getByRange(NDManager,
 * long, long)} or {@link ArrayDataset#getByIndices(NDManager, long...)} to create {@link Batch}
 * instances more efficiently.
 */
public class BulkDataIterable extends DataIterable {

    /**
     * Creates a new instance of {@code BulkDataIterable} with the given parameters.
     *
     * @param dataset the dataset to iterate on
     * @param manager the manager to create the arrays
     * @param sampler a sampler to sample data with
     * @param dataBatchifier a batchifier for data
     * @param labelBatchifier a batchifier for labels
     * @param pipeline the pipeline of transforms to apply on the data
     * @param targetPipeline the pipeline of transforms to apply on the labels
     * @param executor an {@link ExecutorService}
     * @param preFetchNumber the number of samples to prefetch
     * @param device the {@link Device}
     */
    public BulkDataIterable(
            ArrayDataset dataset,
            NDManager manager,
            Sampler sampler,
            Batchifier dataBatchifier,
            Batchifier labelBatchifier,
            Pipeline pipeline,
            Pipeline targetPipeline,
            ExecutorService executor,
            int preFetchNumber,
            Device device) {
        super(
                dataset,
                manager,
                sampler,
                dataBatchifier,
                labelBatchifier,
                pipeline,
                targetPipeline,
                executor,
                preFetchNumber,
                device);
    }

    @Override
    protected Batch fetch(List<Long> indices, int progress) throws IOException {
        NDManager subManager = manager.newSubManager();
        subManager.setName("dataIter fetch");
        int batchSize = indices.size();

        Batch raw;
        if (isRange(indices)) {
            long fromIndex = indices.get(0);
            long toIndex = fromIndex + indices.size();
            raw = ((ArrayDataset) dataset).getByRange(subManager, fromIndex, toIndex);
        } else {
            long[] indicesArr = indices.stream().mapToLong(Long::longValue).toArray();
            raw = ((ArrayDataset) dataset).getByIndices(subManager, indicesArr);
        }

        NDList batchData = raw.getData();
        // apply transform
        if (pipeline != null) {
            batchData = pipeline.transform(batchData);
        }

        NDList batchLabels = raw.getLabels();

        // apply label transform
        if (targetPipeline != null) {
            batchLabels = targetPipeline.transform(batchLabels);
        }
        // pin to a specific device
        if (device != null) {
            batchData = batchData.toDevice(device, false);
            batchLabels = batchLabels.toDevice(device, false);
        }
        return new Batch(
                subManager,
                batchData,
                batchLabels,
                batchSize,
                dataBatchifier,
                labelBatchifier,
                progress,
                dataset.size(),
                indices);
    }

    /**
     * Checks whether the given indices actually represents a range.
     *
     * @param indices the indices to examine
     * @return whether the given indices are sorted in ascending order with no gap and has at least
     *     one element
     */
    public static boolean isRange(List<Long> indices) {
        if (indices.isEmpty()) {
            return false;
        }
        long from = indices.get(0);
        for (long index : indices) {
            if (index != from++) {
                return false;
            }
        }
        return true;
    }
}
