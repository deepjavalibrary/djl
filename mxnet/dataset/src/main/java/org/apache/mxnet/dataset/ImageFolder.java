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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxImages;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.dataset.DataIterable;
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

// TODO add integration test
// TODO put ImageFolder under mxnet for now it should be in Joule-api

/** A dataset for loading image files stored in a folder structure. */
public final class ImageFolder implements RandomAccessDataset {
    private static final String[] EXT =
            new String[] {".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"};
    private static final Logger logger = LoggerFactory.getLogger(ImageFolder.class);

    private NDManager manager;
    private DataLoadingConfiguration config;
    private String root;
    private MxImages.Flag flag;
    private List<String> synsets;
    private PairList<String, Integer> items;

    private ImageFolder(Builder builder) {
        this.manager = builder.manager;
        this.config = builder.config;
        this.root = builder.root;
        this.flag = builder.flag;
        this.synsets = new ArrayList<>();
        this.items = new PairList<>();
        listImage(root);
    }

    @Override
    public Pair<NDList, NDList> get(long index) {
        NDArray data = MxImages.read(manager, items.get(Math.toIntExact(index)).getKey(), flag);
        NDArray label = manager.create(items.get(Math.toIntExact(index)).getValue());
        return new Pair<>(new NDList(data), new NDList(label));
    }

    @Override
    public long size() {
        return items.size();
    }

    @Override
    public Iterable<Record> getRecords() {
        return new DataIterable(this, config);
    }

    private void listImage(String root) {
        File[] dir = new File(root).listFiles();
        if (dir == null || dir.length == 0) {
            throw new IllegalArgumentException(
                    String.format("%s not found or didn't have any file in it", root));
        }
        Arrays.sort(dir);
        for (File file : dir) {
            if (!file.isDirectory()) {
                logger.warn("Ignoring {}, which is not a directory.", file);
                continue;
            }
            int label = synsets.size();
            synsets.add(file.getName());
            File[] images = new File(file.getPath()).listFiles();
            if (images == null || images.length == 0) {
                logger.warn("{} folder is empty", file);
                continue;
            }
            Arrays.sort(images);
            for (File image : images) {
                if (Arrays.stream(EXT)
                        .anyMatch(ext -> image.getName().toLowerCase().endsWith(ext))) {
                    items.add(new Pair<>(image.getPath(), label));
                } else {
                    logger.warn("ImageIO didn't support {} Ignoring... ", image.getName());
                }
            }
        }
    }

    public static final class Builder {
        private NDManager manager;
        private String root;
        private MxImages.Flag flag;
        private DataLoadingConfiguration config;

        public Builder(NDManager manager) {
            this.manager = manager;
        }

        public Builder setPath(String root) {
            this.root = root;
            return this;
        }

        public Builder setFlag(MxImages.Flag flag) {
            this.flag = flag;
            return this;
        }

        public Builder setDataLoadingProperty(boolean shuffle, int batchSize, boolean dropLast) {
            this.config =
                    new DataLoadingConfiguration.Builder()
                            .setShuffle(false)
                            .setBatchSize(batchSize)
                            .setDropLast(dropLast)
                            .build();
            return this;
        }

        public Builder setDataLoadingProperty(DataLoadingConfiguration config) {
            if (this.config != null) {
                throw new IllegalArgumentException(
                        "either setDataLoading or setDataLoadingConfig, not both");
            }
            this.config = config;
            return this;
        }

        public ImageFolder build() {
            if (config == null) {
                this.config =
                        new DataLoadingConfiguration.Builder()
                                .setShuffle(false)
                                .setBatchSize(1)
                                .setDropLast(false)
                                .build();
            }
            if (flag == null) {
                flag = MxImages.Flag.COLOR;
            }
            return new ImageFolder(this);
        }
    }
}
