/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.training.dataset;

import java.util.Iterator;
import java.util.List;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

// TODO abstract a interface that could be inherited by this and Stream DataIterable
// where the random reads is expensive
public class DataIterable<I, L> implements Iterable<Batch> {
    private RandomAccessDataset<I, L> dataset;
    private Trainer<I, L, ?> trainer;
    private Sampler sampler;
    private DataLoadingConfiguration config;

    public DataIterable(
            RandomAccessDataset<I, L> dataset,
            Trainer<I, L, ?> trainer,
            Sampler sampler,
            DataLoadingConfiguration config) {
        this.dataset = dataset;
        this.trainer = trainer;
        this.sampler = sampler;
        this.config = config;
    }

    @Override
    public Iterator<Batch> iterator() {
        return new DataIterator<>(dataset, trainer, sampler, config);
    }

    private static class DataIterator<I, L> implements Iterator<Batch> {
        private RandomAccessDataset<I, L> dataset;
        private Trainer<I, L, ?> trainer;
        private Iterator<List<Long>> sample;
        private Device pinDevice;

        public DataIterator(
                RandomAccessDataset<I, L> dataset,
                Trainer<I, L, ?> trainer,
                Sampler sampler,
                DataLoadingConfiguration config) {
            this.dataset = dataset;
            this.trainer = trainer;
            this.sample = sampler.sample(trainer, dataset);
            this.pinDevice = config.getPinDevice();
        }

        @Override
        public boolean hasNext() {
            return sample.hasNext();
        }

        @Override
        public Batch next() {
            List<Long> indices = sample.next();
            NDList[] data = new NDList[indices.size()];
            NDList[] labels = new NDList[indices.size()];
            TranslatorContext ctx = trainer.getPreprocessContext();
            TrainTranslator<I, L, ?> translator = trainer.getTranslator();
            for (int i = 0; i < indices.size(); i++) {
                Pair<I, L> dataItem = dataset.get(indices.get(i));
                Record record;
                try {
                    record = translator.processInput(ctx, dataItem);
                } catch (Exception e) {
                    throw new IllegalStateException("Failed to get next data item", e);
                }
                data[i] = record.getData();
                labels[i] = record.getLabels();
            }
            Batchifier batchifier = translator.getBatchifier();
            NDList batchData = batchifier.batchify(data);
            NDList batchLabels = batchifier.batchify(labels);
            // pin to a specific device
            if (pinDevice != null) {
                batchData = batchData.asInContext(pinDevice, false);
                batchLabels = batchLabels.asInContext(pinDevice, false);
            }
            Batch batch = new Batch(trainer.getManager(), batchData, batchLabels);
            ctx.close();
            return batch;
        }
    }
}
