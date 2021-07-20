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
 * MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist.
 *
 * <p>Each sample is an image (in 3-D NDArray) with shape (28, 28, 1).
 */
public final class Mnist extends ArrayDataset {

    private static final String ARTIFACT_ID = "mnist";
    private static final String VERSION = "1.0";

    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int NUM_CLASSES = 10;

    private NDManager manager;
    private Usage usage;

    private MRL mrl;
    private boolean prepared;

    private Mnist(Builder builder) {
        super(builder);
        this.manager = builder.manager;
        this.manager.setName("mnist");
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Creates a builder to build a {@link Mnist}.
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
        Artifact.Item imageItem;
        Artifact.Item labelItem;
        switch (usage) {
            case TRAIN:
                imageItem = map.get("train_data");
                labelItem = map.get("train_labels");
                break;
            case TEST:
                imageItem = map.get("test_data");
                labelItem = map.get("test_labels");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        labels = new NDArray[] {readLabel(labelItem)};
        data = new NDArray[] {readData(imageItem, labels[0].size())};
        prepared = true;
    }

    private NDArray readData(Artifact.Item item, long length) throws IOException {
        try (InputStream is = mrl.getRepository().openStream(item, null)) {
            if (is.skip(16) != 16) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(is);
            try (NDArray array = manager.create(new Shape(length, 28, 28, 1), DataType.UINT8)) {
                array.set(buf);
                return array.toType(DataType.FLOAT32, false);
            }
        }
    }

    private NDArray readLabel(Artifact.Item item) throws IOException {
        try (InputStream is = mrl.getRepository().openStream(item, null)) {
            if (is.skip(8) != 8) {
                throw new AssertionError("Failed skip data.");
            }

            byte[] buf = Utils.toByteArray(is);
            try (NDArray array = manager.create(new Shape(buf.length), DataType.UINT8)) {
                array.set(buf);
                return array.toType(DataType.FLOAT32, false);
            }
        }
    }

    /** A builder for a {@link Mnist}. */
    public static final class Builder extends BaseBuilder<Builder> {

        private NDManager manager;
        private Repository repository;
        private String groupId;
        private String artifactId;
        private Usage usage;

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
         * @param manager the manager
         * @return this builder
         */
        public Builder optManager(NDManager manager) {
            this.manager.close();
            this.manager = manager.newSubManager();
            return this;
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
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
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Builds the {@link Mnist}.
         *
         * @return the {@link Mnist}
         */
        public Mnist build() {
            return new Mnist(this);
        }

        MRL getMrl() {
            return repository.dataset(CV.ANY, groupId, artifactId, VERSION);
        }
    }
}
