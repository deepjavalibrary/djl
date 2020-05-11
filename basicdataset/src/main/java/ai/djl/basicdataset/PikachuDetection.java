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

import ai.djl.Application.CV;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.Pipeline;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Pikachu image detection dataset that contains multiple Pikachus in each image. */
public class PikachuDetection extends RandomAccessDataset implements ZooDataset {
    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "pikachu";
    private static final Gson GSON =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .create();

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;
    private Image.Flag flag;

    private List<Path> imagePaths;
    private List<float[]> labels;

    protected PikachuDetection(Builder builder) {
        super(builder);
        repository = builder.repository;
        artifact = builder.artifact;
        usage = builder.usage;
        flag = builder.flag;
        imagePaths = new ArrayList<>();
        labels = new ArrayList<>();
    }

    /**
     * Creates a new builder to build a {@link PikachuDetection}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return MRL.dataset(CV.OBJECT_DETECTION, BasicDatasets.GROUP_ID, ARTIFACT_ID);
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
                usagePath = Paths.get("train");
                break;
            case TEST:
                usagePath = Paths.get("test");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        usagePath = root.resolve(usagePath);
        Path indexFile = usagePath.resolve("index.file");
        try (Reader reader = Files.newBufferedReader(indexFile)) {
            Type mapType = new TypeToken<Map<String, List<Float>>>() {}.getType();
            Map<String, List<Float>> metadata = GSON.fromJson(reader, mapType);
            for (Map.Entry<String, List<Float>> entry : metadata.entrySet()) {
                float[] labelArray = new float[5];
                String imgName = entry.getKey();
                List<Float> label = entry.getValue();
                // Class label
                labelArray[0] = label.get(4);

                // Bounding box labels
                labelArray[1] = label.get(5);
                labelArray[2] = label.get(6);
                labelArray[3] = label.get(7);
                labelArray[4] = label.get(8);
                imagePaths.add(usagePath.resolve(imgName));
                labels.add(labelArray);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        int idx = Math.toIntExact(index);
        NDList d =
                new NDList(
                        ImageFactory.getInstance()
                                .fromFile(imagePaths.get(idx))
                                .toNDArray(manager, flag));
        NDArray label = manager.create(labels.get(idx));
        NDList l = new NDList(label.reshape(new Shape(1).addAll(label.getShape())));
        return new Record(d, l);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return imagePaths.size();
    }

    /** A builder for a {@link PikachuDetection}. */
    public static final class Builder extends BaseBuilder<Builder> {

        Repository repository;
        Artifact artifact;
        Usage usage;
        Image.Flag flag;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
            flag = Image.Flag.COLOR;
            pipeline = new Pipeline(new ToTensor());
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
         * Sets the optional color mode flag.
         *
         * @param flag the color mode flag
         * @return this builder
         */
        public Builder optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
        }

        /**
         * Builds the {@link PikachuDetection}.
         *
         * @return the {@link PikachuDetection}
         */
        public PikachuDetection build() {
            return new PikachuDetection(this);
        }
    }
}
