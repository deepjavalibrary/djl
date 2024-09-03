/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

/** A {@link Translator} that handles mask generation task. */
public class Sam2Translator implements NoBatchifyTranslator<Sam2Input, DetectedObjects> {

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private Pipeline pipeline;

    /** Constructs a {@code Sam2Translator} instance. */
    public Sam2Translator() {
        pipeline = new Pipeline();
        pipeline.add(new Resize(1024, 1024));
        pipeline.add(new ToTensor());
        pipeline.add(new Normalize(MEAN, STD));
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Sam2Input input) throws Exception {
        Image image = input.getImage();
        int width = image.getWidth();
        int height = image.getHeight();
        ctx.setAttachment("width", width);
        ctx.setAttachment("height", height);

        List<Point> points = input.getPoints();
        int numPoints = points.size();
        float[] buf = input.toLocationArray(width, height);

        NDManager manager = ctx.getNDManager();
        NDArray array = image.toNDArray(manager, Image.Flag.COLOR);
        array = pipeline.transform(new NDList(array)).get(0).expandDims(0);
        NDArray locations = manager.create(buf, new Shape(1, numPoints, 2));
        NDArray labels = manager.create(input.getLabels());

        return new NDList(array, locations, labels);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray logits = list.get(0);
        NDArray scores = list.get(1).squeeze(0);
        long best = scores.argMax().getLong();

        int width = (Integer) ctx.getAttachment("width");
        int height = (Integer) ctx.getAttachment("height");

        long[] size = {height, width};
        int mode = Image.Interpolation.BILINEAR.ordinal();
        logits = logits.getNDArrayInternal().interpolation(size, mode, false);
        NDArray masks = logits.gt(0f).squeeze(0);

        float[][] dist = Mask.toMask(masks.get(best).toType(DataType.FLOAT32, true));
        Mask mask = new Mask(0, 0, width, height, dist, true);
        double probability = scores.getFloat(best);

        List<String> classes = Collections.singletonList("");
        List<Double> probabilities = Collections.singletonList(probability);
        List<BoundingBox> boxes = Collections.singletonList(mask);

        return new DetectedObjects(classes, probabilities, boxes);
    }

    /** A class represents the segment anything input. */
    public static final class Sam2Input {

        private Image image;
        private List<Point> points;
        private List<Integer> labels;

        /**
         * Constructs a {@code Sam2Input} instance.
         *
         * @param image the image
         * @param points the locations on the image
         * @param labels the labels for the locations (0: background, 1: foreground)
         */
        public Sam2Input(Image image, List<Point> points, List<Integer> labels) {
            this.image = image;
            this.points = points;
            this.labels = labels;
        }

        /**
         * Returns the image.
         *
         * @return the image
         */
        public Image getImage() {
            return image;
        }

        /**
         * Returns the locations.
         *
         * @return the locations
         */
        public List<Point> getPoints() {
            return points;
        }

        float[] toLocationArray(int width, int height) {
            float[] ret = new float[points.size() * 2];
            int i = 0;
            for (Point point : points) {
                ret[i++] = (float) point.getX() / width * 1024;
                ret[i++] = (float) point.getY() / height * 1024;
            }
            return ret;
        }

        int[][] getLabels() {
            return new int[][] {labels.stream().mapToInt(Integer::intValue).toArray()};
        }

        /**
         * Creates a new {@code Sam2Input} instance with the image and a location.
         *
         * @param url the image url
         * @param x the X of the location
         * @param y the Y of the location
         * @return a new {@code Sam2Input} instance
         * @throws IOException if failed to read image
         */
        public static Sam2Input newInstance(String url, int x, int y) throws IOException {
            Image image = ImageFactory.getInstance().fromUrl(url);
            List<Point> points = Collections.singletonList(new Point(x, y));
            List<Integer> labels = Collections.singletonList(1);
            return new Sam2Input(image, points, labels);
        }

        /**
         * Creates a new {@code Sam2Input} instance with the image and a location.
         *
         * @param path the image file path
         * @param x the X of the location
         * @param y the Y of the location
         * @return a new {@code Sam2Input} instance
         * @throws IOException if failed to read image
         */
        public static Sam2Input newInstance(Path path, int x, int y) throws IOException {
            Image image = ImageFactory.getInstance().fromFile(path);
            List<Point> points = Collections.singletonList(new Point(x, y));
            List<Integer> labels = Collections.singletonList(1);
            return new Sam2Input(image, points, labels);
        }
    }
}
