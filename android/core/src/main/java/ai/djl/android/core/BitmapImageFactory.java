/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.android.core;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * {@code BitmapImageFactory} is the Android implementation of {@link ImageFactory}.
 */
public class BitmapImageFactory implements ImageFactory {

    /** {@inheritDoc} */
    @Override
    public Image fromFile(Path path) throws IOException {
        Bitmap bitmap = BitmapFactory.decodeFile(path.toString());
        if (bitmap == null) {
            throw new IOException("Failed to read image from: " + path);
        }
        return new BitMapWrapper(bitmap);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromUrl(URL url) throws IOException {
            return fromInputStream(url.openStream());
    }

    /** {@inheritDoc} */
    @Override
    public Image fromInputStream(InputStream is) throws IOException {
        Bitmap bitmap = BitmapFactory.decodeStream(is);
        if (bitmap == null) {
            throw new IOException("Failed to read image from input stream");
        }
        return new BitMapWrapper(bitmap);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromImage(Object image) {
        if (!(image instanceof Bitmap)) {
            throw new IllegalArgumentException("only Bitmap allowed");
        }
        return new BitMapWrapper((Bitmap) image);
    }

    static class BitMapWrapper implements Image {
        private Bitmap bitmap;

        BitMapWrapper(Bitmap bitmap) {
            this.bitmap = bitmap;
        }

        /** {@inheritDoc} */
        @Override
        public int getWidth() {
            return bitmap.getWidth();
        }

        /** {@inheritDoc} */
        @Override
        public int getHeight() {
            return bitmap.getHeight();
        }

        /** {@inheritDoc} */
        @Override
        public NDArray toNDArray(NDManager manager, Flag flag) {
            int[] pixels = new int[getWidth() * getHeight()];
            int channel;
            if (flag == Flag.GRAYSCALE) {
                channel = 1;
            } else {
                channel = 3;
            }
            ByteBuffer bb = manager.allocateDirect(channel * getWidth() * getHeight());
            bitmap.getPixels(pixels, 0, getWidth(), 0, 0, getWidth(), getHeight());
            for (int rgb : pixels) {
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                if (flag == Flag.GRAYSCALE) {
                    int gray = (red + green + blue) / 3;
                    bb.put((byte) gray);
                } else {
                    bb.put((byte) red);
                    bb.put((byte) green);
                    bb.put((byte) blue);
                }
            }
            bb.rewind();
            return manager.create(bb, new Shape(getHeight(), getWidth(), channel), DataType.UINT8);
        }

        /** {@inheritDoc} */
        @Override
        public void save(OutputStream os, String type) throws IOException {
            if (!bitmap.compress(Bitmap.CompressFormat.valueOf(type.toUpperCase()), 100, os)) {
                throw new IOException("Cannot save image file to output stream File type " +  type);
            }
        }
    }
}
