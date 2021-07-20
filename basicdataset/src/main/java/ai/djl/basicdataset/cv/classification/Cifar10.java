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
package ai.djl.basicdataset.cv.classification;

import ai.djl.Application.CV;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

/**
 * CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html.
 *
 * <p>Each sample is an image (in 3-D {@link NDArray}) with shape (32, 32, 3).
 */
public final class Cifar10 extends ArrayDataset {

    private static final String ARTIFACT_ID = "cifar10";
    private static final String VERSION = "1.0";

    public static final int IMAGE_WIDTH = 32;
    public static final int IMAGE_HEIGHT = 32;

    public static final float[] NORMALIZE_MEAN = {0.4914f, 0.4822f, 0.4465f};
    public static final float[] NORMALIZE_STD = {0.2023f, 0.1994f, 0.2010f};

    // 3072 = 32 * 32 * 3, i.e. one image size, +1 here is label
    private static final int DATA_AND_LABEL_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3 + 1;

    private NDManager manager;
    private Usage usage;

    private MRL mrl;
    private boolean prepared;

    Cifar10(Builder builder) {
        super(builder);
        this.manager = builder.manager;
        this.manager.setName("cifar10");
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Creates a builder to build a {@link Cifar10}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);

        Map<String, Artifact.Item> map = artifact.getFiles();
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
        data =
                new NDArray[] {
                    dataAndLabels
                            .get(":, 1:")
                            .reshape(-1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
                            .transpose(0, 2, 3, 1)
                };
        labels = new NDArray[] {dataAndLabels.get(":,0")};
        // check if data and labels have the same size
        if (data[0].size(0) != labels[0].size(0)) {
            throw new IOException(
                    "the size of data "
                            + data[0].size(0)
                            + " didn't match with the size of labels "
                            + labels[0].size(0));
        }
        prepared = true;
    }

    private NDArray readData(Artifact.Item item) throws IOException {
        try (InputStream is = mrl.getRepository().openStream(item, null)) {
            byte[] buf = Utils.toByteArray(is);
            int length = buf.length / DATA_AND_LABEL_SIZE;
            try (NDArray array =
                    manager.create(new Shape(length, DATA_AND_LABEL_SIZE), DataType.UINT8)) {
                array.set(buf);
                return array.toType(DataType.FLOAT32, false);
            }
        }
    }

    /** A builder to construct a {@link Cifar10}. */
    public static final class Builder extends BaseBuilder<Builder> {

        NDManager manager;
        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            pipeline = new Pipeline(new ToTensor());
            manager = Engine.getInstance().newBaseManager();
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the optional manager for the dataset (default follows engine default).
         *
         * @param manager the new manager
         * @return this builder
         */
        public Builder optManager(NDManager manager) {
            this.manager.close();
            this.manager = manager.newSubManager();
            return this;
        }

        /**
         * Sets the optional repository for the dataset.
         *
         * @param repository the new repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId}
         * @return this builder
         */
        public Builder optGroupId(String groupId) {
            this.groupId = groupId;
            return this;
        }

        /**
         * Sets the optional artifactId.
         *
         * @param artifactId the artifactId
         * @return this builder
         */
        public Builder optArtifactId(String artifactId) {
            if (artifactId.contains(":")) {
                String[] tokens = artifactId.split(":");
                groupId = tokens[0];
                this.artifactId = tokens[1];
            } else {
                this.artifactId = artifactId;
            }
            return this;
        }

        /**
         * Sets the optional usage for the dataset.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Builds a new {@link Cifar10}.
         *
         * @return the new {@link Cifar10}
         */
        public Cifar10 build() {
            return new Cifar10(this);
        }

        MRL getMrl() {
            return repository.dataset(CV.ANY, groupId, artifactId, VERSION);
        }
    }
}
