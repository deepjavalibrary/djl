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
package ai.djl.mxnet.dataset;

import ai.djl.modality.cv.Rectangle;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.modality.cv.util.NDImageUtils.Flag;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CocoDetection extends RandomAccessDataset implements ZooDataset {

    private static final String ARTIFACT_ID = "coco";

    private NDManager manager;
    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;
    private Flag flag;

    private Path dataDir;
    private CocoUtils coco;
    private List<Path> imagePaths;
    private List<double[][]> labels;

    public CocoDetection(Builder builder) {
        super(builder);
        manager = builder.manager;
        repository = builder.repository;
        artifact = builder.artifact;
        usage = builder.usage;
        flag = builder.flag;
        dataDir = builder.dataDir;
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
    public Record get(long index) throws IOException {
        int idx = Math.toIntExact(index);
        NDList d =
                new NDList(BufferedImageUtils.readFileToArray(manager, imagePaths.get(idx), flag));
        NDList l = new NDList(manager.create(labels.get(idx)));
        return new Record(d, l);
    }

    @Override
    public void prepareData(Usage usage) throws IOException {
        if (dataDir == null) {
            Path cacheDir = getRepository().getCacheDirectory();
            URI resourceUri = getArtifact().getResourceUri();
            dataDir = cacheDir.resolve(resourceUri.getPath());
        }

        Path jsonFile;
        switch (usage) {
            case TRAIN:
                jsonFile = dataDir.resolve("annotations").resolve("instances_train2017.json");
                break;
            case TEST:
                jsonFile = dataDir.resolve("annotations").resolve("instances_val2017.json");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        coco = new CocoUtils(jsonFile);
        coco.prepare();
        List<Long> imageIds = coco.getImageIds();
        for (long id : imageIds) {
            Path imagePath = dataDir.resolve(coco.getRelativeImagePath(id));
            List<double[]> labelOfImageId = getLabels(id);
            if (imagePath != null && !labelOfImageId.isEmpty()) {
                imagePaths.add(imagePath);
                labels.add(labelOfImageId.toArray(new double[0][]));
            }
        }

        // set the size
        size = imagePaths.size();
    }

    private double[] convertRecToList(Rectangle rect) {
        double[] list = new double[5];
        list[0] = rect.getX();
        list[1] = rect.getY();
        list[2] = rect.getWidth();
        list[3] = rect.getHeight();
        return list;
    }

    private List<double[]> getLabels(long imageId) {
        List<Long> annotationIds = coco.getAnnotationIdByImageId(imageId);
        if (annotationIds == null) {
            return Collections.emptyList();
        }

        List<double[]> label = new ArrayList<>();
        for (long annotationId : annotationIds) {
            CocoMetadata.Annotation annotation = coco.getAnnotationById(annotationId);
            Rectangle bBox = annotation.getBoundingBox();
            if (annotation.getArea() > 0) {
                double[] list = convertRecToList(bBox);
                // add the category label
                // map the original one to incremental index
                list[4] = coco.mapCategoryId(annotation.getCategoryId());
                label.add(list);
            }
        }
        return label;
    }

    @SuppressWarnings("rawtypes")
    public static final class Builder extends BaseBuilder<Builder> {
        private NDManager manager;
        private Path dataDir;
        private Flag flag = NDImageUtils.Flag.COLOR;
        private Repository repository = Datasets.REPOSITORY;
        private Artifact artifact;
        private Usage usage;

        @Override
        public Builder self() {
            return this;
        }

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return this;
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

        public Builder optFlag(Flag flag) {
            this.flag = flag;
            return self();
        }

        public Builder optDataDir(String dataDir) {
            this.dataDir = Paths.get(dataDir);
            return self();
        }

        public CocoDetection build() {
            return new CocoDetection(this);
        }
    }
}
