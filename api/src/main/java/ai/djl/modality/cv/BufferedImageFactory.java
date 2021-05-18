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
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;
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
    public Image fromUrl(URL url) throws IOException {
        BufferedImage image = ImageIO.read(url);
        if (image == null) {
            throw new IOException("Failed to read image from: " + url);
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
        } else if (array.getDataType() != DataType.UINT8 && array.getDataType() != DataType.INT8) {
            throw new IllegalArgumentException("Datatype should be INT8 or UINT8");
        }
        if (NDImageUtils.isCHW(shape)) {
            int height = (int) shape.get(1);
            int width = (int) shape.get(2);
            int imageArea = width * height;
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int[] raw = array.toUint8Array();
            IntStream.range(0, imageArea)
                    .parallel()
                    .forEach(
                            ele -> {
                                int x = ele % width;
                                int y = ele / width;
                                int red = raw[ele] & 0xFF;
                                int green = raw[ele + imageArea] & 0xFF;
                                int blue = raw[ele + imageArea * 2] & 0xFF;
                                int rgb = (red << 16) | (green << 8) | blue;
                                image.setRGB(x, y, rgb);
                            });
            return new BufferedImageWrapper(image);
        } else {
            int height = (int) shape.get(0);
            int width = (int) shape.get(1);
            int imageArea = width * height;
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int[] raw = array.toUint8Array();
            IntStream.range(0, imageArea)
                    .parallel()
                    .forEach(
                            ele -> {
                                int x = ele % width;
                                int y = ele / width;
                                int red = raw[ele * 3] & 0xFF;
                                int green = raw[ele * 3 + 1] & 0xFF;
                                int blue = raw[ele * 3 + 2] & 0xFF;
                                int rgb = (red << 16) | (green << 8) | blue;
                                image.setRGB(x, y, rgb);
                            });
            return new BufferedImageWrapper(image);
        }
    }

    protected void save(BufferedImage image, OutputStream os, String type) throws IOException {
        ImageIO.write(image, type, os);
    }

    private class BufferedImageWrapper implements Image {

        private final BufferedImage image;

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
        public Object getWrappedImage() {
            return image;
        }

        /** {@inheritDoc} */
        @Override
        public Image getSubimage(int x, int y, int w, int h) {
            return new BufferedImageWrapper(image.getSubimage(x, y, w, h));
        }

        /** {@inheritDoc} */
        @Override
        public Image duplicate(Type type) {
            BufferedImage newImage =
                    new BufferedImage(image.getWidth(), image.getHeight(), getType(type));
            Graphics2D g = newImage.createGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            return new BufferedImageWrapper(newImage);
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
        public void drawBoundingBoxes(DetectedObjects detections) {
            Graphics2D g = (Graphics2D) image.getGraphics();
            int stroke = 2;
            g.setStroke(new BasicStroke(stroke));
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int imageWidth = image.getWidth();
            int imageHeight = image.getHeight();

            List<DetectedObjects.DetectedObject> list = detections.items();
            for (DetectedObjects.DetectedObject result : list) {
                String className = result.getClassName();
                BoundingBox box = result.getBoundingBox();
                g.setPaint(randomColor().darker());

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
                    Mask mask = (Mask) box;
                    drawMask(image, mask);
                } else if (box instanceof Landmark) {
                    drawLandmarks(image, box);
                }
            }
            g.dispose();
        }

        /** {@inheritDoc} */
        @Override
        public void drawJoints(Joints joints) {
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

        private int getType(Type type) {
            if (type == Type.TYPE_INT_ARGB) {
                return BufferedImage.TYPE_INT_ARGB;
            }
            throw new IllegalArgumentException("the type is not supported!");
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

        private void drawMask(BufferedImage image, Mask mask) {
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
                    float opacity = probDist[xCor][yCor];
                    if (opacity < 0.1) {
                        opacity = 0f;
                    }
                    if (opacity > 0.8) {
                        opacity = 0.8f;
                    }
                    maskImage.setRGB(xCor, yCor, new Color(r, g, b, opacity).darker().getRGB());
                }
            }
            Graphics2D gR = (Graphics2D) image.getGraphics();
            gR.drawImage(maskImage, x, y, null);
            gR.dispose();
        }

        private void drawLandmarks(BufferedImage image, BoundingBox box) {
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
