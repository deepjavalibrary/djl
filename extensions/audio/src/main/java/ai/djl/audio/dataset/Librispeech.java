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

package ai.djl.audio.dataset;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class Librispeech extends SpeechRecognitionDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "librispeech";
    private ArrayList<String> audioPaths = new ArrayList<>();

    /**
     * Creates a new instance of {@link SpeechRecognitionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public Librispeech(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        this.mrl = builder.getMrl();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {
        if (prepared) {
            return;
        }
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Artifact.Item item;
        String subPath;
        switch (usage) {
            case TRAIN:
                item = artifact.getFiles().get("train");
                subPath = "LibriSpeech/test-clean-100";
                break;
            case TEST:
                item = artifact.getFiles().get("test");
                subPath = "LibriSpeech/test-clean";
                break;
            default:
                throw new UnsupportedOperationException("Unsupported usage type.");
        }
        File mainDir = mrl.getRepository().getFile(item, subPath).toFile();
        File[] subDirs = mainDir.listFiles();
        List<String> lineArray = new ArrayList<>();
        for (File subDir : subDirs) {
            File[] subSubDirs = subDir.listFiles();
            String subDirName = subDir.getName();
            for (File subSubDir : subSubDirs) {
                String subSubDirName = subSubDir.getName();
                File transFile =
                        new File(
                                String.format(
                                        "%s/%s-%s.trans.txt",
                                        subSubDir.getAbsolutePath(), subDirName, subSubDirName));
                try (BufferedReader reader = Files.newBufferedReader(transFile.toPath())) {
                    String row;
                    while ((row = reader.readLine()) != null) {
                        if (row.contains(" ")) {
                            String trans = row.substring(row.indexOf(" ") + 1);
                            String label = row.substring(0, row.indexOf(" "));
                            String audioIndex = label.split("-")[2];
                            String audioPath =
                                    String.format(
                                            "%s/%s-%s-%s.flac",
                                            transFile.getAbsolutePath(),
                                            subDirName,
                                            subSubDirName,
                                            audioIndex);
                            audioPaths.add(audioPath);
                            lineArray.add(trans);
                        }
                    }
                }
            }
        }
        targetPreprocess(lineArray);
        prepared = true;
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDList data = new NDList();
        NDList labels = new NDList();
        data.add(sourceAudioData.getPreprocessedData(manager, audioPaths.get((int)index)));
        labels.add(targetTextData.getEmbedding(manager, index));
        return new Record(data, labels);
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    /** A builder to construct a {@link Librispeech} . */
    public static class Builder extends AudioBuilder<Librispeech.Builder> {

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Dataset.Usage.TRAIN;
        }

        /**
         * Builds a new {@link Librispeech} object.
         *
         * @return the new {@link Librispeech} object
         */
        public Librispeech build() {
            return new Librispeech(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.Audio.ANY, groupId, artifactId, VERSION);
        }

        /** {@inheritDoc} */
        @Override
        protected Librispeech.Builder self() {
            return this;
        }
    }
}
