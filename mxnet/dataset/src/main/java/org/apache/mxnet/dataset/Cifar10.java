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
package org.apache.mxnet.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.Utils;

/**
 * CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html Each sample
 * is an image (in 3D NDArray) with shape (32, 32, 3).
 */
public final class Cifar10 extends SimpleDataset {

    private static final String ARTIFACT_ID = "cifar10";
    // 3072 = 32 * 32 * 3, i.e. one image size, +1 here is label
    private static final int DATA_AND_LABEL_SIZE = 32 * 32 * 3 + 1;

    public Cifar10(NDManager manager, Usage usage, DataLoadingConfiguration config) {
        super(manager, Datasets.REPOSITORY, usage, config);
    }

    public Cifar10(
            NDManager manager,
            Repository repository,
            Usage usage,
            DataLoadingConfiguration config) {
        super(manager, repository, usage, config);
    }

    public Cifar10(
            NDManager manager,
            Repository repository,
            Artifact artifact,
            Usage usage,
            DataLoadingConfiguration config) {
        super(manager, repository, artifact, usage, config);
    }

    @Override
    public String getArtifactID() {
        return ARTIFACT_ID;
    }

    @Override
    public Pair<NDList, NDList> get(long index) {
        return new Pair<>(new NDList(getData().get(index)), new NDList(getLabels().get(index)));
    }

    @Override
    public void loadData(Usage usage) throws IOException {
        Map<String, Artifact.Item> map = getArtifact().getFiles();
        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = map.get("data_batch.bin");
                break;
            case TEST:
                item = map.get("test_batch.bin");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        NDArray dataAndLabels = readData(item);
        setData(dataAndLabels.get(":, 1:").reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1));
        setLabels(dataAndLabels.get(":,0"));
        // check if data and labels have the same size
        if (getData().size(0) != getLabels().size(0)) {
            throw new IOException(
                    String.format(
                            "the size of data %d didn't match with the size of labels %d",
                            getData().size(0), getLabels().size(0)));
        }
        setSize(getLabels().size());
    }

    public NDArray readData(Artifact.Item item) throws IOException {
        try (InputStream is = getRepository().openStream(item, null)) {
            byte[] buf = Utils.toByteArray(is);
            int length = buf.length / DATA_AND_LABEL_SIZE;
            try (NDArray array =
                    getManager().create(new Shape(length, DATA_AND_LABEL_SIZE), DataType.UINT8)) {
                array.set(buf);
                return array.asType(DataType.FLOAT32, true);
            }
        }
    }
}
