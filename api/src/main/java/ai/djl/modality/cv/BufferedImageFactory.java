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
package ai.djl.modality.cv;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Landmark;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.imageio.ImageIO;

/** {@code BufferedImageFactory} is the default implementation of {@link ImageFactory}. */
public class BufferedImageFactory extends ImageFactory {

    static {
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disables coffee cup image showing up on macOS
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    /** {@inheritDoc} */
    @Override
    public Image fromFile(Path path) throws IOException {
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IOException("Failed to read image from: " + path);
        }
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromInputStream(InputStream is) throws IOException {
        BufferedImage image = ImageIO.read(is);
        if (image == null) {
            throw new IOException("Failed to read image from input stream");
        }
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromImage(Object image) {
        if (!(image instanceof BufferedImage)) {
            throw new IllegalArgumentException("only BufferedImage allowed");
        }
        return new BufferedImageWrapper((BufferedImage) image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromNDArray(NDArray array) {
        Shape shape = array.getShape();
        if (shape.dimension() == 4) {
            throw new UnsupportedOperationException("Batch is not supported");
        } else if (shape.get(0) == 1 || shape.get(2) == 1) {
            throw new UnsupportedOperationException("Grayscale image is not supported");
        }
        int[] raw = array.toType(DataType.UINT8, false).toUint8Array();
        if (NDImageUtils.isCHW(shape)) {
            int height = (int) shape.get(1);
            int width = (int) shape.get(2);
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int[] pixels = new int[width * height];
            int imageArea = height * width;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int index = h * width + w;
                    int red = raw[index];
                    int green = raw[imageArea + index];
                    int blue = raw[imageArea * 2 + index];
                    pixels[index] = (red << 16) | (green << 8) | blue;
                }
            }
            image.setRGB(0, 0, width, height, pixels, 0, width);
            return new BufferedImageWrapper(image);
        }
        int height = (int) shape.get(0);
        int width = (int) shape.get(1);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] pixels = new int[width * height];
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int index = h * width + w;
                int pos = index * 3;
                int red = raw[pos];
                int green = raw[pos + 1];
                int blue = raw[pos + 2];
                pixels[index] = (red << 16) | (green << 8) | blue;
            }
        }
        image.setRGB(0, 0, width, height, pixels, 0, width);
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromPixels(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        image.setRGB(0, 0, width, height, pixels, 0, width);
        return new BufferedImageWrapper(image);
    }

    protected void save(BufferedImage image, OutputStream os, String type) throws IOException {
        ImageIO.write(image, type, os);
    }

    private class BufferedImageWrapper implements Image {

        private BufferedImage image;

        BufferedImageWrapper(BufferedImage image) {
            this.image = image;
        }

        /** {@inheritDoc} */
        @Override
        public int getWidth() {
            return image.getWidth();
        }

        /** {@inheritDoc} */
        @Override
        public int getHeight() {
            return image.getHeight();
        }

        /** {@inheritDoc} */
        @Override
        public BufferedImage getWrappedImage() {
            return image;
        }

        /** {@inheritDoc} */
        @Override
        public BufferedImageWrapper resize(int width, int height, boolean copy) {
            if (!copy && image.getWidth() == width && image.getHeight() == height) {
                return this;
            }

            java.awt.Image img =
                    image.getScaledInstance(width, height, java.awt.Image.SCALE_SMOOTH);
            BufferedImage scaled = new BufferedImage(width, height, image.getType());
            Graphics2D g2d = scaled.createGraphics();
            g2d.drawImage(img, 0, 0, null);
            g2d.dispose();
            return new BufferedImageWrapper(scaled);
        }

        /** {@inheritDoc} */
        @Override
        public Image getSubImage(int x, int y, int w, int h) {
            return new BufferedImageWrapper(image.getSubimage(x, y, w, h));
        }

        /** {@inheritDoc} */
        @Override
        public Image duplicate() {
            BufferedImage copy =
                    new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
            byte[] sourceData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            byte[] biData = ((DataBufferByte) copy.getRaster().getDataBuffer()).getData();
            System.arraycopy(sourceData, 0, biData, 0, sourceData.length);
            return new BufferedImageWrapper(copy);
        }

        /** {@inheritDoc} */
        @Override
        public Image getMask(int[][] mask) {
            int w = mask[0].length;
            int h = mask.length;
            BufferedImageWrapper resized = resize(w, h, true);
            BufferedImage img = resized.getWrappedImage();
            int[] pixels = new int[w * h];
            int index = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (mask[y][x] != 0) {
                        pixels[index] = img.getRGB(x, y);
                    }
                    index++;
                }
            }
            return fromPixels(pixels, w, h);
        }

        private void convertIdNeeded() {
            if (image.getType() == BufferedImage.TYPE_INT_ARGB) {
                return;
            }

            BufferedImage newImage =
                    new BufferedImage(
                            image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = newImage.createGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            image = newImage;
        }

        /** {@inheritDoc} */
        @Override
        public NDArray toNDArray(NDManager manager, Flag flag) {
            int width = image.getWidth();
            int height = image.getHeight();
            int channel;
            if (flag == Flag.GRAYSCALE) {
                channel = 1;
            } else {
                channel = 3;
            }

            ByteBuffer bb = manager.allocateDirect(channel * height * width);
            if (image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
                int[] data = new int[width * height];
                image.getData().getPixels(0, 0, width, height, data);
                for (int gray : data) {
                    byte b = (byte) gray;
                    bb.put(b);
                    if (flag != Flag.GRAYSCALE) {
                        bb.put(b);
                        bb.put(b);
                    }
                }
            } else {
                // get an array of integer pixels in the default RGB color mode
                int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);
                for (int rgb : pixels) {
                    int red = (rgb >> 16) & 0xFF;
                    int green = (rgb >> 8) & 0xFF;
                    int blue = rgb & 0xFF;

                    if (flag == Flag.GRAYSCALE) {
                        int gray = Math.round(0.299f * red + 0.587f * green + 0.114f * blue);
                        bb.put((byte) gray);
                    } else {
                        bb.put((byte) red);
                        bb.put((byte) green);
                        bb.put((byte) blue);
                    }
                }
            }
            bb.rewind();
            return manager.create(bb, new Shape(height, width, channel), DataType.UINT8);
        }

        /** {@inheritDoc} */
        @Override
        public void save(OutputStream os, String type) throws IOException {
            BufferedImageFactory.this.save(image, os, type);
        }

        /** {@inheritDoc} */
        @Override
        public List<BoundingBox> findBoundingBoxes() {
            // TODO: Add grayscale conversion and use BoundFinder to implement
            throw new UnsupportedOperationException("Not supported for BufferedImage");
        }

        /** {@inheritDoc} */
        @Override
        public void drawBoundingBoxes(DetectedObjects detections) {
            // Make image copy with alpha channel because original image was jpg
            convertIdNeeded();

            Graphics2D g = (Graphics2D) image.getGraphics();
            int stroke = 2;
            g.setStroke(new BasicStroke(stroke));
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int imageWidth = image.getWidth();
            int imageHeight = image.getHeight();

            List<DetectedObjects.DetectedObject> list = detections.items();
            int k = 10;
            Map<String, Integer> classNumberTable = new ConcurrentHashMap<>();
            for (DetectedObjects.DetectedObject result : list) {
                String className = result.getClassName();
                BoundingBox box = result.getBoundingBox();
                if (classNumberTable.containsKey(className)) {
                    g.setPaint(new Color(classNumberTable.get(className)));
                } else {
                    g.setPaint(new Color(k));
                    classNumberTable.put(className, k);
                    k = (k + 100) % 255;
                }

                Rectangle rectangle = box.getBounds();
                int x = (int) (rectangle.getX() * imageWidth);
                int y = (int) (rectangle.getY() * imageHeight);
                g.drawRect(
                        x,
                        y,
                        (int) (rectangle.getWidth() * imageWidth),
                        (int) (rectangle.getHeight() * imageHeight));
                drawText(g, className, x, y, stroke, 4);
                // If we have a mask instead of a plain rectangle, draw tha mask
                if (box instanceof Mask) {
                    drawMask((Mask) box);
                } else if (box instanceof Landmark) {
                    drawLandmarks(box);
                }
            }
            g.dispose();
        }

        /** {@inheritDoc} */
        @Override
        public void drawJoints(Joints joints) {
            // Make image copy with alpha channel because original image was jpg
            convertIdNeeded();

            Graphics2D g = (Graphics2D) image.getGraphics();
            int stroke = 2;
            g.setStroke(new BasicStroke(stroke));

            int imageWidth = image.getWidth();
            int imageHeight = image.getHeight();

            for (Joints.Joint joint : joints.getJoints()) {
                g.setPaint(randomColor().darker());
                int x = (int) (joint.getX() * imageWidth);
                int y = (int) (joint.getY() * imageHeight);
                g.fillOval(x, y, 10, 10);
            }
            g.dispose();
        }

        /** {@inheritDoc} */
        @Override
        public void drawImage(Image overlay, boolean resize) {
            if (!(overlay.getWrappedImage() instanceof BufferedImage)) {
                throw new IllegalArgumentException("Only BufferedImage allowed");
            }
            if (resize) {
                overlay = overlay.resize(getWidth(), getHeight(), false);
            }
            BufferedImage target =
                    new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = (Graphics2D) target.getGraphics();
            g.drawImage(image, 0, 0, null);
            g.drawImage((BufferedImage) overlay.getWrappedImage(), 0, 0, null);
            g.dispose();
            image = target;
        }

        private Color randomColor() {
            return new Color(RandomUtils.nextInt(255));
        }

        private void drawText(Graphics2D g, String text, int x, int y, int stroke, int padding) {
            FontMetrics metrics = g.getFontMetrics();
            x += stroke / 2;
            y += stroke / 2;
            int width = metrics.stringWidth(text) + padding * 2 - stroke / 2;
            int height = metrics.getHeight() + metrics.getDescent();
            int ascent = metrics.getAscent();
            java.awt.Rectangle background = new java.awt.Rectangle(x, y, width, height);
            g.fill(background);
            g.setPaint(Color.WHITE);
            g.drawString(text, x + padding, y + ascent);
        }

        private void drawMask(Mask mask) {
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

            BufferedImage maskImage =
                    new BufferedImage(
                            probDist.length, probDist[0].length, BufferedImage.TYPE_INT_ARGB);
            for (int xCor = 0; xCor < probDist.length; xCor++) {
                for (int yCor = 0; yCor < probDist[xCor].length; yCor++) {
                    float opacity = probDist[xCor][yCor] * 0.8f;
                    maskImage.setRGB(xCor, yCor, new Color(r, g, b, opacity).darker().getRGB());
                }
            }
            Graphics2D gR = (Graphics2D) image.getGraphics();
            gR.drawImage(maskImage, x, y, null);
            gR.dispose();
        }

        private void drawLandmarks(BoundingBox box) {
            Graphics2D g = (Graphics2D) image.getGraphics();
            g.setColor(new Color(246, 96, 0));
            BasicStroke bStroke = new BasicStroke(4, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
            g.setStroke(bStroke);
            for (Point point : box.getPath()) {
                g.drawRect((int) point.getX(), (int) point.getY(), 2, 2);
            }
            g.dispose();
        }
    }
}
