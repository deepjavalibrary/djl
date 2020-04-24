/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.Record;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code TatoebaEnglishFrenchDataset} is a English-French machine translation dataset from The
 * Tatoeba Project (http://www.manythings.org/anki/).
 */
public class TatoebaEnglishFrenchDataset extends TextDataset implements ZooDataset {
    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "tatoeba-en-fr";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    /**
     * Creates a new instance of {@code TatoebaEnglishFrenchDataset}.
     *
     * @param builder the builder object to build from
     */
    protected TatoebaEnglishFrenchDataset(Builder builder) {
        super(builder);
        this.repository = builder.repository;
        this.artifact = builder.artifact;
        this.usage = builder.usage;
    }

    /**
     * Creates a new builder to build a {@link TatoebaEnglishFrenchDataset}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return MRL.dataset(
                Application.NLP.MACHINE_TRANSLATION, BasicDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** {@inheritDoc} */
    @Override
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    @Override
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), VERSION, null);
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) throws IOException {
        Path root = repository.getResourceDirectory(artifact);

        Path usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = Paths.get("fra-eng-train.txt");
                break;
            case TEST:
                usagePath = Paths.get("fra-eng-test.txt");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        usagePath = root.resolve(usagePath);

        List<String> sourceTextData = new ArrayList<>();
        List<String> targetTextData = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(usagePath)) {
            String row;
            while ((row = reader.readLine()) != null) {
                String[] text = row.split("\t");
                sourceTextData.add(text[0]);
                targetTextData.add(text[1]);
            }
        }
        try {
            preprocess(sourceTextData, true);
            preprocess(targetTextData, false);
        } catch (EmbeddingException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    @Override
    public Record get(NDManager manager, long index) {
        NDList data = new NDList();
        NDList labels = new NDList();
        data.add(sourceTextData.getEmbedding(manager, index));
        labels.add(targetTextData.getEmbedding(manager, index));

        return new Record(data, labels);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return sourceTextData.getSize();
    }

    /** A builder for a {@link TatoebaEnglishFrenchDataset}. */
    public static class Builder extends TextDataset.Builder<Builder> {
        private Repository repository;
        private Artifact artifact;
        private Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
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
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }

        /**
         * Builds the {@link TatoebaEnglishFrenchDataset}.
         *
         * @return the {@link TatoebaEnglishFrenchDataset}
         */
        public TatoebaEnglishFrenchDataset build() {
            return new TatoebaEnglishFrenchDataset(this);
        }
    }
}
