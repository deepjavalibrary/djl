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
package ai.djl.basicdataset;

import ai.djl.Application;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.PreparedDataset;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.translate.Pipeline;
import ai.djl.util.Progress;
import ai.djl.util.Utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public final class FashionMnist extends ArrayDataset implements PreparedDataset {

    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int NUM_CLASSES = 10;

    private static final String ARTIFACT_ID = "fashmnist";

    private NDManager manager;
    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    /**
     * Creates a new instance of {@code ArrayDataset} with the arguments in {@link Builder}.
     *
     * @param builder a builder with the required arguments
     */
    private FashionMnist(FashionMnist.Builder builder){
        super(builder);
        this.manager = builder.manager;
        this.repository = builder.repository;
        this.artifact = builder.artifact;
        this.usage = builder.usage;
    }

    public static FashionMnist.Builder builder() {
        return new FashionMnist.Builder();
    }

    public MRL getMrl() {
        return MRL.dataset(Application.CV.IMAGE_CLASSIFICATION, BasicDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** {@inheritDoc} */
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), "1.0", null);
    }

    @Override
    public void prepare() throws IOException {
        Map<String, Artifact.Item> map = artifact.getFiles();
        Artifact.Item imageItem;
        Artifact.Item labelItem;
        System.out.println(map);
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
        labels = new NDArray[]{readLabel(labelItem)};
        data = new NDArray[]{readData(imageItem, labels[0].size())};
    }

    @Override
    public void prepare(Progress progress) throws IOException {

    }

    private NDArray readData(Artifact.Item item, long length) throws IOException {
        try (InputStream is = repository.openStream(item, null)) {
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
        try (InputStream is = repository.openStream(item, null)) {
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

    /**
     * A builder for a {@link Mnist}.
     */
    public static final class Builder extends BaseBuilder<FashionMnist.Builder> {

        private NDManager manager;
        private Repository repository;
        private Artifact artifact;
        private Usage usage;

        /**
         * Constructs a new builder.
         */
        Builder() {
            String FASH_MNIST_URL = "src/test/resources/mlrepo/";
            repository = Repository.newInstance("BasicDataset", FASH_MNIST_URL);
            usage = Usage.TRAIN;
            pipeline = new Pipeline(new ToTensor());
            manager = Engine.getInstance().newBaseManager();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected FashionMnist.Builder self() {
            return this;
        }

        /**
         * Sets the optional manager for the dataset (default follows engine default).
         *
         * @param manager the manager
         * @return this builder
         */
        public FashionMnist.Builder optManager(NDManager manager) {
            this.manager = manager.newSubManager();
            return this;
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public FashionMnist.Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public FashionMnist.Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public FashionMnist.Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Builds the {@link Mnist}.
         *
         * @return the {@link Mnist}
         */
        public FashionMnist build() {
            return new FashionMnist(this);
        }
    }

}
