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
package ai.djl.basicdataset.nlp;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The Penn Treebank (PTB) project selected 2,499 stories from a three year Wall Street Journal
 * (WSJ) collection of 98,732 stories for syntactic annotation (see <a
 * href="https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html">here</a> for details).
 */
public class PennTreebankText extends TextDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "penntreebank-unlabeled-processed";

    /**
     * Creates a new instance of {@link PennTreebankText} with the given necessary configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    PennTreebankText(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Creates a builder to build a {@link PennTreebankText}.
     *
     * @return a new {@link PennTreebankText.Builder} object
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDList data = new NDList();
        NDList labels = null;
        data.add(sourceTextData.getEmbedding(manager, index));
        return new Record(data, labels);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return sourceTextData.getSize();
    }

    /**
     * Prepares the dataset for use with tracked progress.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     */
    @Override
    public void prepare(Progress progress) throws IOException, EmbeddingException {
        if (prepared) {
            return;
        }
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = artifact.getFiles().get("train");
                break;
            case TEST:
                item = artifact.getFiles().get("test");
                break;
            case VALIDATION:
                item = artifact.getFiles().get("valid");
                break;
            default:
                throw new UnsupportedOperationException("Unsupported usage type.");
        }
        Path path = mrl.getRepository().getFile(item, "").toAbsolutePath();
        List<String> lineArray = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String row;
            while ((row = reader.readLine()) != null) {
                lineArray.add(row);
            }
        }
        preprocess(lineArray, true);
        prepared = true;
    }

    /** A builder to construct a {@link PennTreebankText} . */
    public static class Builder extends TextDataset.Builder<Builder> {

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Dataset.Usage.TRAIN;
        }

        /**
         * Builds a new {@link PennTreebankText} object.
         *
         * @return the new {@link PennTreebankText} object
         */
        public PennTreebankText build() {
            return new PennTreebankText(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.NLP.ANY, groupId, artifactId, VERSION);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }
}
