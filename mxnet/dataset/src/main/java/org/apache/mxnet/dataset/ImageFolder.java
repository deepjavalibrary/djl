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
import org.apache.mxnet.engine.MxImages;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.dataset.Record;

/** A dataset for loading image files stored in a folder structure. */
public final class ImageFolder extends AbstractImageFolder {

    String root;

    public ImageFolder(Builder builder) {
        super(builder);
        this.root = builder.getRoot();
    }

    @Override
    public Record get(long index) {
        int idx = Math.toIntExact(index);
        NDList d = new NDList(MxImages.read(manager, items.get(idx).getKey(), flag));
        NDList l = new NDList(manager.create(items.get(idx).getValue()));
        return new Record(d, l);
    }

    @Override
    protected NDList readImage(String image) throws IOException {
        return new NDList(MxImages.read(manager, image, flag));
    }

    @Override
    public void prepare() throws IOException {
        listImages(root);
    }

    public static final class Builder extends AbstractImageFolder.BaseBuilder<Builder> {

        private String root;

        public String getRoot() {
            return root;
        }

        public Builder setRoot(String root) {
            this.root = root;
            return self();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public ImageFolder build() {
            return new ImageFolder(this);
        }
    }
}
