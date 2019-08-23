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
import software.amazon.ai.training.dataset.DataLoadingConfiguration;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

// TODO add integration test
// TODO put ImageFolder under mxnet for now it should be in Joule-api

/** A dataset for loading image files stored in a folder structure. */
public final class ImageFolder extends RandomAccessDataset {
    private static final String[] EXT = {".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"};
    private static final Logger logger = LoggerFactory.getLogger(ImageFolder.class);

    private NDManager manager;
    private MxImages.Flag flag;
    private List<String> synsets;
    private PairList<String, Integer> items;

    public ImageFolder(NDManager manager, String root, DataLoadingConfiguration config) {
        this(manager, root, MxImages.Flag.COLOR, config);
    }

    public ImageFolder(
            NDManager manager, String root, MxImages.Flag flag, DataLoadingConfiguration config) {
        super(config);
        this.manager = manager;
        this.flag = flag;
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
}
