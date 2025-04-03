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
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;

/** {@code BitmapImageFactory} is the Android implementation of {@link ImageFactory}. */
public class BitmapImageFactory extends ImageFactory {

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

    /** {@inheritDoc} */
    @Override
    public Image fromNDArray(NDArray array) {
        Shape shape = array.getShape();
        if (shape.dimension() != 3) {
            throw new IllegalArgumentException("Shape should only have three dimension follow CHW");
        } else if (shape.get(0) == 1 || shape.get(2) == 1) {
            throw new UnsupportedOperationException("Grayscale image is not supported");
        }
        int[] raw = array.toType(DataType.UINT8, false).toUint8Array();
        if (NDImageUtils.isCHW(shape)) {
            int height = (int) shape.get(1);
            int width = (int) shape.get(2);
            Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            int imageArea = width * height;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int index = h * width + w;
                    int red = raw[index] & 0xFF;
                    int green = raw[imageArea + index] & 0xFF;
                    int blue = raw[imageArea * 2 + index] & 0xFF;
                    bitmap.setPixel(w, h, Color.argb(255, red, green, blue));
                }
            }
            return new BitMapWrapper(bitmap);
        }
        int height = (int) shape.get(0);
        int width = (int) shape.get(1);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int pos = (h * width + w) * 3;
                int red = raw[pos];
                int green = raw[pos + 1];
                int blue = raw[pos + 2];
                bitmap.setPixel(w, h, Color.argb(255, red, green, blue));
            }
        }
        return new BitMapWrapper(bitmap);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromPixels(int[] pixels, int width, int height) {
        Bitmap  bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        return new BitMapWrapper(bitmap);
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
        public Bitmap getWrappedImage() {
            return bitmap;
        }

        /** {@inheritDoc} */
        @Override
        public BitMapWrapper resize(int width, int height, boolean copy) {
            if (!copy && bitmap.getWidth() == width && bitmap.getHeight() == height) {
                return this;
            }
            return new BitMapWrapper(Bitmap.createScaledBitmap(bitmap, width, height, true));
        }

        /** {@inheritDoc} */
        @Override
        public Image getMask(int[][] mask) {
            int w = mask[0].length;
            int h = mask.length;
            BitMapWrapper resized = resize(w, h, true);
            Bitmap img = resized.getWrappedImage();
            int[] pixels = new int[w * h];
            int index = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (mask[y][x] != 0) {
                        pixels[index] = img.getPixel(x, y);
                    }
                    index++;
                }
            }
            return new BitMapWrapper(Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888));
        }

        /** {@inheritDoc} */
        @Override
        public Image getSubImage(int x, int y, int w, int h) {
            return new BitMapWrapper(Bitmap.createBitmap(bitmap, x, y, w, h));
        }

        /** {@inheritDoc} */
        @Override
        public Image duplicate() {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            return new BitMapWrapper(mutableBitmap);
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
                throw new IOException("Cannot save image file to output stream File type " + type);
            }
        }

        /** {@inheritDoc} */
        @Override
        public List<BoundingBox> findBoundingBoxes() {
            // TODO: Add grayscale conversion and use BoundFinder to implement
            throw new UnsupportedOperationException("Not supported for BitMapImage");
        }

        /** {@inheritDoc} */
        @Override
        public void drawBoundingBoxes(DetectedObjects detections, float opacity) {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);
            // set the paint configure
            int stroke = 2;
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(stroke);
            paint.setAntiAlias(true);

            int imageWidth = mutableBitmap.getWidth();
            int imageHeight = mutableBitmap.getHeight();

            List<DetectedObjects.DetectedObject> list = detections.items();
            for (DetectedObjects.DetectedObject result : list) {
                String className = result.getClassName();
                BoundingBox box = result.getBoundingBox();
                int color = darker(randomColor());
                paint.setColor(color);

                Rectangle rectangle = box.getBounds();
                int x = (int) (rectangle.getX() * imageWidth);
                int y = (int) (rectangle.getY() * imageHeight);
                canvas.drawRect(
                        x,
                        y,
                        x + (int) (rectangle.getWidth() * imageWidth),
                        y + (int) (rectangle.getHeight() * imageHeight),
                        paint);
                drawText(canvas, color, className, x, y, stroke, 4);
                // If we have a mask instead of a plain rectangle, draw tha mask
                if (box instanceof Mask) {
                    Mask mask = (Mask) box;
                    drawMask(mutableBitmap, mask);
                }
            }
            Bitmap oldBitmap = bitmap;
            bitmap = mutableBitmap;
            oldBitmap.recycle();
        }

        /** {@inheritDoc} */
        @Override
        public void drawRectangle(Rectangle rect, int rgb, int thickness) {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);
            int color = darker(rgb);

            // set the paint configure
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(thickness);
            paint.setAntiAlias(true);
            paint.setColor(color);

            int imageWidth = mutableBitmap.getWidth();
            int imageHeight = mutableBitmap.getHeight();
            int x = (int) (rect.getX() * imageWidth);
            int y = (int) (rect.getY() * imageHeight);
            canvas.drawRect(
                    x,
                    y,
                    x + (int) (rect.getWidth() * imageWidth),
                    y + (int) (rect.getHeight() * imageHeight),
                    paint);
            Bitmap oldBitmap = bitmap;
            bitmap = mutableBitmap;
            oldBitmap.recycle();
        }

        /** {@inheritDoc} */
        @Override
        public void drawMarks(List<Point> points, int radius) {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);

            // set the paint configure
            Paint paint = new Paint();
            paint.setStrokeWidth(2);
            paint.setStyle(Paint.Style.FILL);
            paint.setAntiAlias(true);

            paint.setColor(darker(randomColor()));
            for (Point point : points) {
                int[][] star = createStar(point, radius);
                drawPolygon(canvas, paint, star);
            }

            Bitmap oldBitmap = bitmap;
            bitmap = mutableBitmap;
            oldBitmap.recycle();
        }

        /** {@inheritDoc} */
        @Override
        public void drawJoints(Joints joints) {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableBitmap);
            // set the paint configure
            Paint paint = new Paint();
            paint.setStrokeWidth(2);
            paint.setStyle(Paint.Style.FILL);
            paint.setAntiAlias(true);

            int imageWidth = mutableBitmap.getWidth();
            int imageHeight = mutableBitmap.getHeight();

            paint.setColor(darker(randomColor()));
            for (Joints.Joint joint : joints.getJoints()) {
                int x = (int) (joint.getX() * imageWidth);
                int y = (int) (joint.getY() * imageHeight);
                canvas.drawOval(x, y, x + 10, y + 10, paint);
            }
            Bitmap oldBitmap = bitmap;
            bitmap = mutableBitmap;
            oldBitmap.recycle();
        }

        /** {@inheritDoc} */
        @Override
        public void drawImage(Image overlay, boolean resize) {
            if (!(overlay.getWrappedImage() instanceof Bitmap)) {
                throw new IllegalArgumentException("Only Bitmap allowed");
            }
            if (resize) {
                overlay = overlay.resize(getWidth(), getHeight(), false);
            }
            Bitmap target = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(target);
            canvas.drawBitmap(bitmap, 0, 0, null);
            canvas.drawBitmap((Bitmap) overlay.getWrappedImage(), 0, 0, null);
            Bitmap oldBitmap = bitmap;
            bitmap = target;
            oldBitmap.recycle();
        }

        private int randomColor() {
            return Color.rgb(
                    RandomUtils.nextInt(256), RandomUtils.nextInt(256), RandomUtils.nextInt(256));
        }

        private int darker(int color) {
            int a = Color.alpha(color);
            int r = Math.round(Color.red(color) * 0.8f);
            int g = Math.round(Color.green(color) * 0.8f);
            int b = Math.round(Color.blue(color) * 0.8f);
            return Color.argb(a, Math.min(r, 255), Math.min(g, 255), Math.min(b, 255));
        }

        private void drawText(
                Canvas canvas, int color, String text, int x, int y, int stroke, int padding) {
            Paint paint = new Paint();
            Paint.FontMetrics metrics = paint.getFontMetrics();
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(color);
            paint.setAntiAlias(true);

            x += stroke / 2;
            y += stroke / 2;

            int width = (int) (paint.measureText(text) + padding * 2 - stroke / 2);
            // the height here includes ascent
            int height = (int) (metrics.descent - metrics.ascent);
            int ascent = (int) metrics.ascent;
            Rect bounds = new Rect(x, y, x + width, y + height);
            canvas.drawRect(bounds, paint);
            paint.setColor(Color.WHITE);
            // ascent in android is negative value, so y = y - ascent
            canvas.drawText(text, x + padding, y - ascent, paint);
        }

        private void drawMask(Bitmap image, Mask mask) {
            float r = RandomUtils.nextFloat();
            float g = RandomUtils.nextFloat();
            float b = RandomUtils.nextFloat();
            int imageWidth = image.getWidth();
            int imageHeight = image.getHeight();
            int x = (int) (mask.getX() * imageWidth);
            int y = (int) (mask.getY() * imageHeight);
            float[][] probDist = mask.getProbDist();
            // Correct some coordinates of box when going out of image
            if (x < 0) {
                x = 0;
            }
            if (y < 0) {
                y = 0;
            }

            Bitmap maskedImage =
                    Bitmap.createBitmap(
                            probDist.length, probDist[0].length, Bitmap.Config.ARGB_8888);
            for (int xCor = 0; xCor < probDist.length; xCor++) {
                for (int yCor = 0; yCor < probDist[xCor].length; yCor++) {
                    float opacity = probDist[xCor][yCor];
                    if (opacity < 0.1) {
                        opacity = 0f;
                    }
                    if (opacity > 0.8) {
                        opacity = 0.8f;
                    }
                    maskedImage.setPixel(xCor, yCor, darker(Color.argb(opacity, r, g, b)));
                }
            }
            Canvas canvas = new Canvas(image);
            canvas.drawBitmap(maskedImage, x, y, null);
        }

        private void drawPolygon(Canvas canvas, Paint paint, int[][] polygon) {
            android.graphics.Path polygonPath = new android.graphics.Path();
            polygonPath.moveTo(polygon[0][0], polygon[1][0]);
            for (int i = 1; i < polygon[0].length; i++) {
                polygonPath.lineTo(polygon[0][i], polygon[1][i]);
            }
            polygonPath.close();
            canvas.drawPath(polygonPath, paint);
        }
    }
}
