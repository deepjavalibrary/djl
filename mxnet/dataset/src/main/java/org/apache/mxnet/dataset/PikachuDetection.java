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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.modality.cv.util.NDImageUtils.Flag;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;

public class PikachuDetection extends RandomAccessDataset implements ZooDataset {
    private static final String ARTIFACT_ID = "pikachu";
    private static final Gson GSON =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .create();

    private NDManager manager;
    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;
    private Flag flag = NDImageUtils.Flag.COLOR;

    private Path dataDir;
    private List<Path> imagePaths;
    private List<float[]> labels;

    public PikachuDetection(Builder builder) {
        super(builder);
        manager = builder.manager;
        repository = builder.repository;
        artifact = builder.artifact;
        dataDir = builder.dataDir;
        usage = builder.usage;
        imagePaths = new ArrayList<>();
        labels = new ArrayList<>();
    }

    @Override
    public MRL getMrl() {
        return new MRL(MRL.Dataset.CV, Datasets.GROUP_ID, ARTIFACT_ID);
    }

    @Override
    public Repository getRepository() {
        return repository;
    }

    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    @Override
    public Usage getUsage() {
        return usage;
    }

    @Override
    public boolean isPrepared() {
        return prepared;
    }

    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), "1.0", null);
    }

    @Override
    public void prepare() throws IOException {
        if (isPrepared()) {
            return;
        }
        // it uses local files not need to download
        if (dataDir != null) {
            prepareData(usage);
            setPrepared(true);
            return;
        }
        // download the dataset from remote
        ZooDataset.super.prepare();
    }

    @Override
    public void prepareData(Usage usage) throws IOException {
        if (dataDir == null) {
            setDataDir();
        }
        Path usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = Paths.get("train");
                break;
            case TEST:
                usagePath = Paths.get("test");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        usagePath = dataDir.resolve(usagePath);
        Path indexFile = usagePath.resolve("index.file");
        try (Reader reader = Files.newBufferedReader(indexFile)) {
            Type mapType = new TypeToken<Map<String, List<Float>>>() {}.getType();
            Map<String, List<Float>> metadata = GSON.fromJson(reader, mapType);
            for (Map.Entry<String, List<Float>> entry : metadata.entrySet()) {
                float[] labelArray = new float[5];
                String imgName = entry.getKey();
                List<Float> label = entry.getValue();
                // Offset labels
                labelArray[0] = label.get(5);
                labelArray[1] = label.get(6);
                labelArray[2] = label.get(7);
                labelArray[3] = label.get(8);

                // Class label
                labelArray[4] = label.get(4);
                imagePaths.add(usagePath.resolve(imgName));
                labels.add(labelArray);
            }
        }
        size = imagePaths.size();
    }

    private void setDataDir() throws IOException {
        Path cacheDir = getRepository().getCacheDirectory();
        URI resourceUri = getArtifact().getResourceUri();
        dataDir = cacheDir.resolve(resourceUri.getPath());
    }

    @Override
    public Record get(long index) throws IOException {
        int idx = Math.toIntExact(index);
        NDList d =
                new NDList(
                        BufferedImageUtils.readFileToArray(manager, imagePaths.get(idx), flag)
                                .transpose(2, 0, 1));
        NDArray label = manager.create(labels.get(idx));
        NDList l = new NDList(label.reshape(new Shape(1).addAll(label.getShape())));
        return new Record(d, l);
    }

    public static final class Builder extends BaseBuilder<Builder> {

        private Path dataDir;
        private Repository repository = Datasets.REPOSITORY;
        private Artifact artifact;
        private Usage usage;
        private NDManager manager;

        @Override
        public Builder self() {
            return this;
        }

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return self();
        }

        public Builder setUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        public Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }

        public Builder optDataDir(String dataDir) {
            this.dataDir = Paths.get(dataDir);
            return self();
        }

        public PikachuDetection build() {
            return new PikachuDetection(this);
        }
    }
}
