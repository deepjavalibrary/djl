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

import com.google.gson.Gson;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.repository.NakedRepository;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.util.PairList;

public class ImageNet extends AbstractImageFolder implements PreparedDataset {

    private Repository repository;
    private Usage usage;
    private boolean prepared;
    private Path root;

    private String[] wordNetIds;
    private String[] classNames;
    private String[] classFull;

    public ImageNet(Builder builder) {
        super(builder);
        this.repository = builder.repository;
        this.usage = builder.usage;
        this.synsets = new ArrayList<>();
        this.items = new PairList<>();
    }

    public String[] getWordNetIds() {
        return wordNetIds;
    }

    public String[] getClassNames() {
        return classNames;
    }

    public String[] getClassFull() {
        return classFull;
    }

    @Override
    public void prepare() throws IOException {
        if (!prepared) {
            prepareClasses();
            prepareItems();
            prepared = true;
        }
    }

    private void prepareClasses() {
        InputStream classStream =
                Thread.currentThread()
                        .getContextClassLoader()
                        .getResourceAsStream("imagenet/classes.json");
        String[][] classes =
                new Gson()
                        .fromJson(
                                new InputStreamReader(classStream, StandardCharsets.UTF_8),
                                String[][].class);
        wordNetIds = new String[classes.length];
        classNames = new String[classes.length];
        classFull = new String[classes.length];
        for (int i = 0; i < classes.length; i++) {
            wordNetIds[i] = classes[i][0];
            classNames[i] = classes[i][1];
            classFull[i] = classes[i][2];
        }
    }

    private void prepareItems() throws IOException {
        String usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = "train";
                break;
            case VALIDATION:
                usagePath = "val";
                break;
            case TEST:
                throw new UnsupportedOperationException("Test data not available.");
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
        root = Paths.get(repository.getBaseUri()).resolve(usagePath);
        listImages(root.toString(), wordNetIds);
    }

    @Override
    protected NDList readImage(String image) throws IOException {
        Path imagePath = root.resolve(image);
        return new NDList(BufferedImageUtils.readFileToArray(manager, imagePath, flag));
    }

    public static class Builder extends AbstractImageFolder.BaseBuilder<Builder> {

        private Repository repository = Datasets.REPOSITORY;
        private Usage usage;

        public Builder optRepository(Repository repository) {
            if (!(repository instanceof NakedRepository)) {
                throw new IllegalArgumentException("ImageNet requires a NakedRepository");
            }
            this.repository = repository;
            return this;
        }

        public Builder setUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        @Override
        public Builder self() {
            return this;
        }

        public ImageNet build() {
            return new ImageNet(this);
        }
    }
}
