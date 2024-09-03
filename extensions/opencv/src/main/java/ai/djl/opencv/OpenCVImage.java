/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.opencv;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Landmark;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/** {@code OpenCVImage} is a high performance implementation of {@link Image}. */
class OpenCVImage implements Image {

    private Mat image;

    /**
     * Constructs a new {@code OpenCVImage} instance.
     *
     * @param image the wrapped image
     */
    public OpenCVImage(Mat image) {
        this.image = image;
    }

    /** {@inheritDoc} */
    @Override
    public int getWidth() {
        return image.width();
    }

    /** {@inheritDoc} */
    @Override
    public int getHeight() {
        return image.height();
    }

    /** {@inheritDoc} */
    @Override
    public Mat getWrappedImage() {
        return image;
    }

    /** {@inheritDoc} */
    @Override
    public OpenCVImage resize(int width, int height, boolean copy) {
        if (!copy && image.width() == width && image.height() == height) {
            return this;
        }

        Mat resized = new Mat();
        Imgproc.resize(image, resized, new Size(width, height));
        return new OpenCVImage(resized);
    }

    /** {@inheritDoc} */
    @Override
    public Image getMask(int[][] mask) {
        int w = mask[0].length;
        int h = mask.length;
        OpenCVImage resized = resize(w, h, false);
        Mat img = resized.getWrappedImage();
        Mat ret = new Mat(h, w, CvType.CV_8UC4);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (mask[y][x] != 0) {
                    double[] data = img.get(y, x);
                    ret.put(y, x, data[0], data[1], data[2], 255);
                }
            }
        }
        return new OpenCVImage(ret);
    }

    /** {@inheritDoc} */
    @Override
    public Image getSubImage(int x, int y, int w, int h) {
        Mat mat = image.submat(new Rect(x, y, w, h));
        return new OpenCVImage(mat);
    }

    /** {@inheritDoc} */
    @Override
    public Image duplicate() {
        Mat mat = new Mat();
        image.copyTo(mat);
        return new OpenCVImage(mat);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toNDArray(NDManager manager, Flag flag) {
        Mat mat = new Mat();
        if (flag == Flag.GRAYSCALE) {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2GRAY);
        } else {
            Imgproc.cvtColor(image, mat, Imgproc.COLOR_BGR2RGB);
        }
        byte[] buf = new byte[mat.height() * mat.width() * mat.channels()];
        mat.get(0, 0, buf);
        Shape shape = new Shape(mat.height(), mat.width(), mat.channels());
        return manager.create(ByteBuffer.wrap(buf), shape, DataType.UINT8);
    }

    /** {@inheritDoc} */
    @Override
    public void save(OutputStream os, String type) throws IOException {
        MatOfByte buf = new MatOfByte();
        if (!Imgcodecs.imencode('.' + type, image, buf)) {
            throw new IOException("Failed save image.");
        }
        os.write(buf.toArray());
    }

    /** {@inheritDoc} */
    @Override
    public void drawBoundingBoxes(DetectedObjects detections, float opacity) {
        int imageWidth = image.width();
        int imageHeight = image.height();

        List<DetectedObjects.DetectedObject> list = detections.items();
        for (DetectedObjects.DetectedObject result : list) {
            String className = result.getClassName();
            BoundingBox box = result.getBoundingBox();

            Rectangle rectangle = box.getBounds();
            int x = (int) (rectangle.getX() * imageWidth);
            int y = (int) (rectangle.getY() * imageHeight);
            Rect rect =
                    new Rect(
                            x,
                            y,
                            (int) (rectangle.getWidth() * imageWidth),
                            (int) (rectangle.getHeight() * imageHeight));
            Scalar color =
                    new Scalar(
                            RandomUtils.nextInt(178),
                            RandomUtils.nextInt(178),
                            RandomUtils.nextInt(178));
            Imgproc.rectangle(image, rect.tl(), rect.br(), color, 2);

            Size size = Imgproc.getTextSize(className, Imgproc.FONT_HERSHEY_PLAIN, 1.3, 1, null);
            Point br = new Point(x + size.width + 4, y + size.height + 4);
            Imgproc.rectangle(image, rect.tl(), br, color, -1);

            Point point = new Point(x, y + size.height + 2);
            color = new Scalar(255, 255, 255);
            Imgproc.putText(image, className, point, Imgproc.FONT_HERSHEY_PLAIN, 1.3, color, 1);
            // If we have a mask instead of a plain rectangle, draw tha mask
            if (box instanceof Mask) {
                Mask mask = (Mask) box;
                BufferedImage img = mat2Image(image);
                drawMask(img, mask, 0.5f);
                image = image2Mat(img);
            } else if (box instanceof Landmark) {
                drawLandmarks(box);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void drawMarks(List<ai.djl.modality.cv.output.Point> points, int radius) {
        Scalar color = new Scalar(190, 150, 37);
        for (ai.djl.modality.cv.output.Point point : points) {
            int[][] star = createStar(point, radius);
            Point[] mat = new Point[10];
            for (int i = 0; i < 10; ++i) {
                mat[i] = new Point(star[0][i], star[1][i]);
            }
            MatOfPoint mop = new MatOfPoint();
            mop.fromArray(mat);
            List<MatOfPoint> ppt = Collections.singletonList(mop);
            Imgproc.fillPoly(image, ppt, color, Imgproc.LINE_AA);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void drawJoints(Joints joints) {
        int imageWidth = image.width();
        int imageHeight = image.height();

        List<Joints.Joint> list = joints.getJoints();
        if (list.size() == 17) {
            Scalar color = new Scalar(37, 255, 224);
            drawLine(list.get(5), list.get(7), imageWidth, imageHeight, color);
            drawLine(list.get(7), list.get(9), imageWidth, imageHeight, color);
            drawLine(list.get(6), list.get(8), imageWidth, imageHeight, color);
            drawLine(list.get(8), list.get(10), imageWidth, imageHeight, color);
            drawLine(list.get(11), list.get(13), imageWidth, imageHeight, color);
            drawLine(list.get(12), list.get(14), imageWidth, imageHeight, color);
            drawLine(list.get(13), list.get(15), imageWidth, imageHeight, color);
            drawLine(list.get(14), list.get(16), imageWidth, imageHeight, color);
            drawLine(list.get(5), list.get(6), imageWidth, imageHeight, color);
            drawLine(list.get(11), list.get(12), imageWidth, imageHeight, color);
            drawLine(list.get(5), list.get(11), imageWidth, imageHeight, color);
            drawLine(list.get(6), list.get(12), imageWidth, imageHeight, color);
        }

        Scalar color = new Scalar(190, 150, 37);
        for (Joints.Joint joint : list) {
            int x = (int) (joint.getX() * imageWidth);
            int y = (int) (joint.getY() * imageHeight);
            Point point = new Point(x, y);
            Imgproc.circle(image, point, 6, color, -1, Imgproc.LINE_AA);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void drawImage(Image overlay, boolean resize) {
        if (!(overlay instanceof OpenCVImage)) {
            throw new IllegalArgumentException("Only OpenCVImage allowed");
        }
        if (resize) {
            overlay = overlay.resize(getWidth(), getHeight(), false);
        }
        Mat mat = (Mat) overlay.getWrappedImage();
        if (mat.elemSize() != 4) {
            mat.copyTo(image);
            return;
        }
        int w = Math.min(image.width(), mat.width());
        int h = Math.min(image.height(), mat.height());
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                /*
                 * RA = SA + DA × (1 − SA)
                 * R[0] = (S[0]×SA + D[0]×DA×(1 − SA)) / RA
                 */
                double[] src = mat.get(y, x);
                double[] dest = image.get(y, x);
                double sa = src[3];
                double da;
                double ra;
                if (dest.length == 3) {
                    da = 255 - sa;
                    ra = 255;
                } else {
                    da = dest[3] * (255 - sa) / 255;
                    ra = sa + da;
                    dest[3] = ra;
                }

                dest[0] = (src[0] * sa + dest[0] * da) / ra;
                dest[1] = (src[1] * sa + dest[1] * da) / ra;
                dest[2] = (src[2] * sa + dest[2] * da) / ra;
                image.put(y, x, dest);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<BoundingBox> findBoundingBoxes() {
        List<MatOfPoint> points = new ArrayList<>();
        Imgproc.findContours(
                image, points, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        return points.parallelStream()
                .map(
                        point -> {
                            Rect rect = Imgproc.boundingRect(point);
                            return new Rectangle(
                                    rect.x * 1.0 / image.width(),
                                    rect.y * 1.0 / image.height(),
                                    rect.width * 1.0 / image.width(),
                                    rect.height * 1.0 / image.height());
                        })
                .collect(Collectors.toList());
    }

    /**
     * Converting from bgr mapping to rgb.
     *
     * @return rgb format image
     */
    public OpenCVImage bgr2rgb() {
        Mat converted = new Mat();
        Imgproc.cvtColor(image, converted, Imgproc.COLOR_BGR2RGB);
        return new OpenCVImage(converted);
    }

    /**
     * Converting from channel-first to channel-last format.
     *
     * @return channel last image
     */
    public OpenCVImage chw2hwc() {
        int c = image.channels();
        int h = image.height();
        int w = image.width();

        Mat cHW = image.reshape(0, new int[] {c, h * w});
        Mat result = new Mat();
        result.create(h, w, CvType.makeType(image.depth(), c));
        result = result.reshape(c, new int[] {h, w});
        Core.transpose(cHW, result);
        return new OpenCVImage(result);
    }

    /**
     * Converting from channel-las tto channel-first format.
     *
     * @return channel first image
     */
    public OpenCVImage hwc2chw() {
        int c = image.channels();
        int h = image.height();
        int w = image.width();
        Mat hWC = image.reshape(1, h * w);
        Mat result = new Mat();
        Core.transpose(hWC, result);
        result = result.reshape(1, new int[] {c, h, w});
        return new OpenCVImage(result);
    }

    /**
     * Apply normalization on the image.
     *
     * @param mean mean value apply on each color channel
     * @param std standard div apply on each color channel
     * @return converted image
     */
    public OpenCVImage normalize(float[] mean, float[] std) {
        Mat result = new Mat();
        Core.subtract(image, new Scalar(mean[0], mean[1], mean[2]), result);
        Core.divide(result, new Scalar(std[0], std[1], std[2]), result);
        return new OpenCVImage(result);
    }

    private void drawLine(Joints.Joint from, Joints.Joint to, int width, int height, Scalar color) {
        int x0 = (int) (from.getX() * width);
        int y0 = (int) (from.getY() * height);
        int x1 = (int) (to.getX() * width);
        int y1 = (int) (to.getY() * height);
        Imgproc.line(image, new Point(x0, y0), new Point(x1, y1), color, 2, Imgproc.LINE_AA);
    }

    private void drawLandmarks(BoundingBox box) {
        Scalar color = new Scalar(0, 96, 246);
        for (ai.djl.modality.cv.output.Point point : box.getPath()) {
            Point lt = new Point(point.getX() - 4, point.getY() - 4);
            Point rb = new Point(point.getX() + 4, point.getY() + 4);
            Imgproc.rectangle(image, lt, rb, color, -1);
        }
    }

    private void drawMask(BufferedImage img, Mask mask, float ratio) {
        // TODO: use OpenCV native way to draw mask
        float r = RandomUtils.nextFloat();
        float g = RandomUtils.nextFloat();
        float b = RandomUtils.nextFloat();
        int imageWidth = img.getWidth();
        int imageHeight = img.getHeight();
        int x = 0;
        int y = 0;
        int w = imageWidth;
        int h = imageHeight;
        if (!mask.isFullImageMask()) {
            x = (int) (mask.getX() * imageWidth);
            y = (int) (mask.getY() * imageHeight);
            w = (int) (mask.getWidth() * imageWidth);
            h = (int) (mask.getHeight() * imageHeight);
            // Correct some coordinates of box when going out of image
            if (x < 0) {
                x = 0;
            }
            if (y < 0) {
                y = 0;
            }
        }
        float[][] probDist = mask.getProbDist();
        if (ratio < 0 || ratio > 1) {
            float max = 0;
            for (float[] row : probDist) {
                for (float f : row) {
                    max = Math.max(max, f);
                }
            }
            ratio = 0.5f / max;
        }

        BufferedImage maskImage =
                new BufferedImage(probDist[0].length, probDist.length, BufferedImage.TYPE_INT_ARGB);
        for (int yCor = 0; yCor < probDist.length; yCor++) {
            for (int xCor = 0; xCor < probDist[0].length; xCor++) {
                float opacity = probDist[yCor][xCor] * ratio;
                maskImage.setRGB(xCor, yCor, new Color(r, g, b, opacity).darker().getRGB());
            }
        }
        java.awt.Image scaled = maskImage.getScaledInstance(w, h, java.awt.Image.SCALE_SMOOTH);

        Graphics2D gR = (Graphics2D) img.getGraphics();
        gR.drawImage(scaled, x, y, null);
        gR.dispose();
    }

    private static BufferedImage mat2Image(Mat mat) {
        int width = mat.width();
        int height = mat.height();
        byte[] data = new byte[width * height * (int) mat.elemSize()];
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);

        mat.get(0, 0, data);

        BufferedImage ret = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        ret.getRaster().setDataElements(0, 0, width, height, data);
        return ret;
    }

    private static Mat image2Mat(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        byte[] data;
        Mat mat;
        DataBuffer buf = img.getRaster().getDataBuffer();
        if (buf instanceof DataBufferByte) {
            data = ((DataBufferByte) buf).getData();
            mat = new Mat(height, width, CvType.CV_8UC3);
        } else if (buf instanceof DataBufferInt) {
            int[] intData = ((DataBufferInt) buf).getData();
            data = new byte[intData.length * 4];
            ByteBuffer bb = ByteBuffer.wrap(data);
            bb.asIntBuffer().put(intData);
            mat = new Mat(height, width, CvType.CV_8UC4);
        } else {
            throw new IllegalArgumentException("Unsupported image type: " + buf.getClass());
        }
        mat.put(0, 0, data);
        return mat;
    }
}
