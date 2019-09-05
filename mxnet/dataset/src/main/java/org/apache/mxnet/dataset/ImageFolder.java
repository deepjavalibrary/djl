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
import org.apache.mxnet.engine.MxImages.Flag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.dataset.RandomAccessDataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

// TODO add integration test
// TODO put ImageFolder under mxnet for now it should be in Joule-api

/** A dataset for loading image files stored in a folder structure. */
public class ImageFolder extends RandomAccessDataset<String, Integer> {
    private static final String[] EXT = {".jpg", ".jpeg", ".png", ".bmp", ".wbmp", ".gif"};
    private static final Logger logger = LoggerFactory.getLogger(ImageFolder.class);

    private MxImages.Flag flag;
    private List<String> synsets;
    private PairList<String, Integer> items;

    public ImageFolder(BaseBuilder<?> builder) {
        super(builder);
        this.flag = builder.getFlag();
        this.synsets = new ArrayList<>();
        this.items = new PairList<>();
        listImage(builder.getRoot());
    }

    @Override
    public Pair<String, Integer> get(long index) {
        return items.get(Math.toIntExact(index));
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

    public DefaultTranslator defaultTranslator() {
        return new DefaultTranslator();
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<B extends BaseBuilder>
            extends RandomAccessDataset.BaseBuilder<B> {

        private MxImages.Flag flag = Flag.COLOR;
        private String root;

        public Flag getFlag() {
            return flag;
        }

        public B optFlag(Flag flag) {
            this.flag = flag;
            return self();
        }

        public String getRoot() {
            return root;
        }

        public B setRoot(String root) {
            this.root = root;
            return self();
        }
    }

    public static class Builder extends BaseBuilder<Builder> {

        @Override
        public Builder self() {
            return this;
        }

        public ImageFolder build() {
            return new ImageFolder(this);
        }
    }

    private class DefaultTranslator implements TrainTranslator<String, Integer, NDList> {

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) throws Exception {
            return null;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) throws Exception {
            return new NDList(MxImages.read(ctx.getNDManager(), input, flag));
        }

        @Override
        public Record processInput(TranslatorContext ctx, String input, Integer label)
                throws Exception {
            NDList i = new NDList(MxImages.read(ctx.getNDManager(), input, flag));
            NDList l = new NDList(ctx.getNDManager().create(label));
            return new Record(i, l);
        }
    }
}
