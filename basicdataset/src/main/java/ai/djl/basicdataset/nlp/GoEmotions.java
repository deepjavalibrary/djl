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
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/**
 * GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human
 * annotations to 27 emotion categories or Neutral. This version of data is filtered based on
 * rater-agreement on top of the raw data, and contains a train/test/validation split. The emotion
 * categories are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
 * desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief,
 * joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.
 */
public class GoEmotions extends TextDataset {

    private static final String ARTIFACT_ID = "goemotions";
    private static final String VERSION = "1.0";

    List<int[]> targetData = new ArrayList<>();

    enum HeaderEnum {
        text,
        emotion_id,
        comment_id
    }

    /**
     * Creates a new instance of {@link GoEmotions}.
     *
     * @param builder the builder object to build from
     */
    GoEmotions(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Prepares the dataset for use with tracked progress. In this method the TSV file will be
     * parsed. All datasets will be preprocessed.
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
        Path root = mrl.getRepository().getResourceDirectory(artifact);

        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("train.tsv");
                break;
            case TEST:
                csvFile = root.resolve("test.tsv");
                break;
            case VALIDATION:
                csvFile = root.resolve("dev.tsv");
                break;
            default:
                throw new UnsupportedOperationException("Data not available.");
        }

        CSVFormat csvFormat =
                CSVFormat.TDF.builder().setQuote(null).setHeader(HeaderEnum.class).build();
        URL csvUrl = csvFile.toUri().toURL();
        List<CSVRecord> csvRecords;
        List<String> sourceTextData = new ArrayList<>();

        try (Reader reader =
                new InputStreamReader(
                        new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8)) {
            CSVParser csvParser = new CSVParser(reader, csvFormat);
            csvRecords = csvParser.getRecords();
        }

        for (CSVRecord csvRecord : csvRecords) {
            sourceTextData.add(csvRecord.get(0));
            String[] labels = csvRecord.get(1).split(",");
            int[] labelInt = new int[labels.length];
            for (int i = 0; i < labels.length; i++) {
                labelInt[i] = Integer.parseInt(labels[i]);
            }
            targetData.add(labelInt);
        }

        preprocess(sourceTextData, true);
        prepared = true;
    }

    /**
     * Gets the {@link Record} for the given index from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param index the index of the requested data item
     * @return a {@link Record} that contains the data and label of the requested data item. The
     *     data {@link NDList} contains three {@link NDArray}s representing the embedded title,
     *     context and question, which are named accordingly. The label {@link NDList} contains
     *     multiple {@link NDArray}s corresponding to each embedded answer.
     */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDList data = new NDList();
        NDList labels = new NDList();
        data.add(sourceTextData.getEmbedding(manager, index));
        labels.add(manager.create(targetData.get((int) index)));

        return new Record(data, labels);
    }

    /**
     * Returns the number of records available to be read in this {@code Dataset}. In this
     * implementation, the actual size of available records are the size of {@code
     * questionInfoList}.
     *
     * @return the number of records available to be read in this {@code Dataset}
     */
    @Override
    protected long availableSize() {
        return sourceTextData.getSize();
    }

    /**
     * Creates a builder to build a {@link GoEmotions}.
     *
     * @return a new builder
     */
    public static GoEmotions.Builder builder() {
        return new GoEmotions.Builder();
    }

    /** A builder to construct a {@link GoEmotions}. */
    public static final class Builder extends TextDataset.Builder<GoEmotions.Builder> {

        /** Constructs a new builder. */
        public Builder() {
            artifactId = ARTIFACT_ID;
        }

        /** {@inheritDoc} */
        @Override
        public GoEmotions.Builder self() {
            return this;
        }

        /**
         * Builds the {@link TatoebaEnglishFrenchDataset}.
         *
         * @return the {@link TatoebaEnglishFrenchDataset}
         */
        public GoEmotions build() {
            return new GoEmotions(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.NLP.ANY, groupId, artifactId, VERSION);
        }
    }
}
