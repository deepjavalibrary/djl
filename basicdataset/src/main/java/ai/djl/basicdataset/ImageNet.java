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

import ai.djl.repository.dataset.PreparedDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Progress;
import com.google.gson.Gson;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class ImageNet extends AbstractImageFolder implements PreparedDataset {

    private Usage usage;
    private boolean prepared;
    private String[] wordNetIds;
    private String[] classNames;
    private String[] classFull;

    ImageNet(Builder builder) {
        super(builder);
        this.usage = builder.usage;
    }

    public static ImageFolder.Builder builder() {
        return new ImageFolder.Builder();
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

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (!prepared) {
            if (progress != null) {
                progress.reset("Preparing", 2);
                prepareClasses();
                progress.start(0);
                prepareItems();
                progress.end();
            } else {
                prepareClasses();
                prepareItems();
            }

            prepared = true;
        }
    }

    private void prepareClasses() {
        InputStream classStream =
                Thread.currentThread()
                        .getContextClassLoader()
                        .getResourceAsStream("imagenet/classes.json");
        if (classStream == null) {
            throw new AssertionError("Missing imagenet/classes.json in jar resource");
        }
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

    /** {@inheritDoc} */
    @Override
    public List<String> getSynset() {
        if (!prepared) {
            throw new IllegalStateException("Please call prepare() first");
        }
        return Arrays.asList(classNames);
    }

    private String getUsagePath(Dataset.Usage usage) {
        String usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = "train";
                return usagePath;
            case VALIDATION:
                usagePath = "val";
                return usagePath;
            case TEST:
                throw new UnsupportedOperationException("Test data not available.");
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
    }

    private void prepareItems() throws IOException {
        Path root = Paths.get(repository.getBaseUri()).resolve(getUsagePath(usage));
        listImages(root.toString(), wordNetIds);
    }

    /** {@inheritDoc} */
    @Override
    protected Path getImagePath(String key) {
        return Paths.get(repository.getBaseUri()).resolve(getUsagePath(usage)).resolve(key);
    }

    public static class Builder extends ImageFolderBuilder<Builder> {

        private Usage usage = Usage.TRAIN;

        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }

        public ImageNet build() {
            return new ImageNet(this);
        }
    }
}
