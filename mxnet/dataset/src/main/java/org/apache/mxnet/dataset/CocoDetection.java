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

import java.io.IOException;
import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.engine.MxImages;
import software.amazon.ai.modality.cv.Rectangle;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

public class CocoDetection extends RandomAccessDataset<String, double[][]>
        implements ZooDataset<String, double[][]> {

    private static final String ARTIFACT_ID = "coco";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;
    private MxImages.Flag flag;

    private String dataDir;
    private List<String> imagePaths;
    private List<double[][]> labels;
    private List<Long> imageIds;
    private Map<Long, CocoMetadata.Image> imageMap;
    private Map<Long, CocoMetadata.Annotation> annotationMap;
    private Map<Long, List<Long>> imageToAnn;
    private Map<Long, Integer> categoryIdMap;

    public CocoDetection(Builder builder) {
        super(builder);
        repository = builder.repository;
        artifact = builder.artifact;
        usage = builder.usage;
        flag = builder.flag;
        dataDir = builder.dataDir;
        imageIds = new ArrayList<>();
        imageMap = new HashMap<>();
        annotationMap = new HashMap<>();
        imageToAnn = new HashMap<>();
        categoryIdMap = new HashMap<>();
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
    public Pair<String, double[][]> get(long index) {
        int idx = Math.toIntExact(index);
        return new Pair<>(imagePaths.get(idx), labels.get(idx));
    }

    @Override
    public void prepareData(Usage usage) throws IOException {
        if (dataDir == null) {
            setDataDir();
        }
        // load the json into memory
        createIndex(usage);

        imagePaths = new ArrayList<>(imageIds.size());
        labels = new ArrayList<>(imageIds.size());
        for (long id : imageIds) {
            String imagePath = getImagePath(id);
            List<double[]> labelOfImageId = getLabels(id);
            if (imagePath != null && !labelOfImageId.isEmpty()) {
                imagePaths.add(imagePath);
                labels.add(labelOfImageId.toArray(new double[0][]));
            }
        }

        // set the size
        size = imagePaths.size();
    }

    private void setDataDir() throws IOException {
        Path cacheDir = getRepository().getCacheDirectory();
        URI resourceUri = getArtifact().getResourceUri();
        Path resourceDir = cacheDir.resolve(resourceUri.getPath());
        dataDir = resourceDir.toString();
    }

    private void createIndex(Usage usage) throws IOException {
        Path jsonFile;
        switch (usage) {
            case TRAIN:
                jsonFile = Paths.get("annotations", "instances_train2017.json");
                break;
            case TEST:
                jsonFile = Paths.get("annotations", "instances_val2017.json");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }

        Path cocoPath = Paths.get(dataDir);
        CocoMetadata metadata;
        try (Reader reader = Files.newBufferedReader(cocoPath.resolve(jsonFile))) {
            metadata = CocoMetadata.GSON.fromJson(reader, CocoMetadata.class);
        }
        for (CocoMetadata.Annotation annotation : metadata.getAnnotations()) {
            long imageId = annotation.getImageId();
            long id = annotation.getId();
            if (!imageToAnn.containsKey(imageId)) {
                imageToAnn.put(annotation.getImageId(), new ArrayList<>());
            }
            imageToAnn.get(imageId).add(id);
            annotationMap.put(id, annotation);
        }

        for (CocoMetadata.Image image : metadata.getImages()) {
            imageIds.add(image.getId());
            imageMap.put(image.getId(), image);
        }

        // create categoryIndex
        List<Long> categoryIds = new ArrayList<>();
        for (CocoMetadata.Category category : metadata.getCategories()) {
            categoryIds.add(category.getId());
        }
        for (int i = 0; i < categoryIds.size(); i++) {
            categoryIdMap.put(categoryIds.get(i), i);
        }
        // sort to keep the dataset ordered
        Collections.sort(imageIds);
    }

    private String getImagePath(long imageId) {
        CocoMetadata.Image image = imageMap.get(imageId);
        String[] cocoUrl = image.getCocoUrl().split("/");
        Path path = Paths.get(dataDir);
        return path.resolve(Paths.get(cocoUrl[cocoUrl.length - 2]))
                .resolve(Paths.get(cocoUrl[cocoUrl.length - 1]))
                .toString();
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
        List<Long> annotationIds = imageToAnn.get(imageId);
        if (annotationIds == null) {
            return Collections.emptyList();
        }

        List<double[]> label = new ArrayList<>();
        for (long annotationId : annotationIds) {
            CocoMetadata.Annotation annotation = annotationMap.get(annotationId);
            Rectangle bBox = annotation.getBoundingBox();
            if (annotation.getArea() > 0) {
                double[] list = convertRecToList(bBox);
                // add the category label
                // map the original one to incremental index
                list[4] = categoryIdMap.get(annotation.getCategoryId());
                label.add(list);
            }
        }
        return label;
    }

    public DefaultTranslator defaultTranslator() {
        return new DefaultTranslator();
    }

    @SuppressWarnings("rawtypes")
    public static final class Builder extends BaseBuilder<Builder> {

        private String dataDir;
        private MxImages.Flag flag = MxImages.Flag.COLOR;
        private Repository repository = Datasets.REPOSITORY;
        private Artifact artifact;
        private Usage usage;

        @Override
        public Builder self() {
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

        public Builder optFlag(MxImages.Flag flag) {
            this.flag = flag;
            return self();
        }

        public Builder optDataDir(String dataDir) {
            this.dataDir = dataDir;
            return self();
        }

        public CocoDetection build() {
            return new CocoDetection(this);
        }
    }

    private class DefaultTranslator implements TrainTranslator<String, double[][], NDList> {

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            return null;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            return new NDList(MxImages.read(ctx.getNDManager(), input, flag));
        }

        @Override
        public Record processInput(TranslatorContext ctx, String input, double[][] label) {
            NDList i = new NDList(MxImages.read(ctx.getNDManager(), input, flag));
            NDList l = new NDList(ctx.getNDManager().create(label));
            return new Record(i, l);
        }
    }
}
