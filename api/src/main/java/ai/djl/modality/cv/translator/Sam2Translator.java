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

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import com.google.gson.annotations.SerializedName;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** A {@link Translator} that handles mask generation task. */
public class Sam2Translator implements NoBatchifyTranslator<Sam2Input, DetectedObjects> {

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private Pipeline pipeline;
    private Predictor<NDList, NDList> predictor;
    private String encoderPath;
    private String encodeMethod;

    /** Constructs a {@code Sam2Translator} instance. */
    public Sam2Translator(Builder builder) {
        pipeline = new Pipeline();
        pipeline.add(new Resize(1024, 1024));
        pipeline.add(new ToTensor());
        pipeline.add(new Normalize(MEAN, STD));
        this.encoderPath = builder.encoderPath;
        this.encodeMethod = builder.encodeMethod;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException, ModelException {
        if (encoderPath == null) {
            // PyTorch model
            if (encodeMethod != null) {
                Model model = ctx.getModel();
                predictor = model.newPredictor(new NoopTranslator(null));
                model.getNDManager().attachInternal(NDManager.nextUid(), predictor);
            }
            return;
        }
        Model model = ctx.getModel();
        Path path = Paths.get(encoderPath);
        if (!path.isAbsolute() && Files.notExists(path)) {
            path = model.getModelPath().resolve(encoderPath);
        }
        if (!Files.exists(path)) {
            throw new IOException("encoder model not found: " + encoderPath);
        }
        NDManager manager = ctx.getNDManager();
        Model encoder = manager.getEngine().newModel("encoder", manager.getDevice());
        encoder.load(path);
        predictor = encoder.newPredictor(new NoopTranslator(null));
        model.getNDManager().attachInternal(NDManager.nextUid(), predictor);
        model.getNDManager().attachInternal(NDManager.nextUid(), encoder);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Sam2Input input) throws Exception {
        Image image = input.getImage();
        int width = image.getWidth();
        int height = image.getHeight();
        ctx.setAttachment("width", width);
        ctx.setAttachment("height", height);

        float[] buf = input.toLocationArray(width, height);

        NDManager manager = ctx.getNDManager();
        NDArray array = image.toNDArray(manager, Image.Flag.COLOR);
        array = pipeline.transform(new NDList(array)).get(0).expandDims(0);
        NDArray locations = manager.create(buf, new Shape(1, buf.length / 2, 2));
        NDArray labels = manager.create(input.getLabels());

        if (predictor == null) {
            return new NDList(array, locations, labels);
        }

        NDList embeddings;
        if (encodeMethod == null) {
            embeddings = predictor.predict(new NDList(array));
        } else {
            NDArray placeholder = manager.create("");
            placeholder.setName("module_method:" + encodeMethod);
            embeddings = predictor.predict(new NDList(placeholder, array));
        }

        NDArray mask = manager.zeros(new Shape(1, 1, 256, 256));
        NDArray hasMask = manager.zeros(new Shape(1));
        for (NDArray arr : embeddings) {
            arr.setName(null);
        }
        return new NDList(
                embeddings.get(2),
                embeddings.get(0),
                embeddings.get(1),
                locations,
                labels,
                mask,
                hasMask);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
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

    /**
     * Creates a builder to build a {@code Sam2Translator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return builder(Collections.emptyMap());
    }

    /**
     * Creates a builder to build a {@code Sam2Translator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        return new Builder(arguments);
    }

    /** The builder for Sam2Translator. */
    public static class Builder {

        String encoderPath;
        String encodeMethod;

        Builder(Map<String, ?> arguments) {
            encoderPath = ArgumentsUtil.stringValue(arguments, "encoder");
            encodeMethod = ArgumentsUtil.stringValue(arguments, "encode_method");
        }

        /**
         * Sets the encoder model path.
         *
         * @param encoderPath the encoder model path
         * @return the builder
         */
        public Builder optEncoderPath(String encoderPath) {
            this.encoderPath = encoderPath;
            return this;
        }

        /**
         * Sets the module name for encode method.
         *
         * @param encodeMethod the module name for encode method
         * @return the builder
         */
        public Builder optEncodeMethod(String encodeMethod) {
            this.encodeMethod = encodeMethod;
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public Sam2Translator build() {
            return new Sam2Translator(this);
        }
    }

    /** A class represents the segment anything input. */
    public static final class Sam2Input {

        private Image image;
        private Point[] points;
        private int[] labels;
        private boolean visualize;

        /**
         * Constructs a {@code Sam2Input} instance.
         *
         * @param image the image
         * @param points the locations on the image
         * @param labels the labels for the locations (0: background, 1: foreground)
         */
        public Sam2Input(Image image, Point[] points, int[] labels) {
            this(image, points, labels, false);
        }

        /**
         * Constructs a {@code Sam2Input} instance.
         *
         * @param image the image
         * @param points the locations on the image
         * @param labels the labels for the locations (0: background, 1: foreground)
         * @param visualize true if output visualized image
         */
        public Sam2Input(Image image, Point[] points, int[] labels, boolean visualize) {
            this.image = image;
            this.points = points;
            this.labels = labels;
            this.visualize = visualize;
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
         * Returns {@code true} if output visualized image.
         *
         * @return {@code true} if output visualized image
         */
        public boolean isVisualize() {
            return visualize;
        }

        /**
         * Returns the locations.
         *
         * @return the locations
         */
        public List<Point> getPoints() {
            List<Point> list = new ArrayList<>();
            for (int i = 0; i < labels.length; ++i) {
                if (labels[i] < 2) {
                    list.add(points[i]);
                }
            }
            return list;
        }

        /**
         * Returns the box.
         *
         * @return the box
         */
        public List<Rectangle> getBoxes() {
            List<Rectangle> list = new ArrayList<>();
            for (int i = 0; i < labels.length; ++i) {
                if (labels[i] == 2) {
                    double width = points[i + 1].getX() - points[i].getX();
                    double height = points[i + 1].getY() - points[i].getY();
                    list.add(new Rectangle(points[i], width, height));
                }
            }
            return list;
        }

        float[] toLocationArray(int width, int height) {
            float[] ret = new float[points.length * 2];
            int i = 0;
            for (Point point : points) {
                ret[i++] = (float) point.getX() / width * 1024;
                ret[i++] = (float) point.getY() / height * 1024;
            }
            return ret;
        }

        float[][] getLabels() {
            float[][] buf = new float[1][labels.length];
            for (int i = 0; i < labels.length; ++i) {
                buf[0][i] = labels[i];
            }
            return buf;
        }

        /**
         * Constructs a {@code Sam2Input} instance from json string.
         *
         * @param input the json input
         * @return a {@code Sam2Input} instance
         * @throws IOException if failed to load the image
         */
        public static Sam2Input fromJson(String input) throws IOException {
            Prompt prompt = JsonUtils.GSON.fromJson(input, Prompt.class);
            if (prompt.image == null) {
                throw new IllegalArgumentException("Missing image_url value");
            }
            if (prompt.prompt == null || prompt.prompt.length == 0) {
                throw new IllegalArgumentException("Missing prompt value");
            }
            Image image = ImageFactory.getInstance().fromUrl(prompt.image);
            Builder builder = builder(image);
            if (prompt.visualize) {
                builder.visualize();
            }
            for (Location location : prompt.prompt) {
                int[] data = location.data;
                if ("point".equals(location.type)) {
                    builder.addPoint(data[0], data[1], location.label);
                } else if ("rectangle".equals(location.type)) {
                    builder.addBox(data[0], data[1], data[2], data[3]);
                }
            }
            return builder.build();
        }

        /**
         * Creates a builder to build a {@code Sam2Input} with the image.
         *
         * @param image the image
         * @return a new builder
         */
        public static Builder builder(Image image) {
            return new Builder(image);
        }

        /** The builder for {@code Sam2Input}. */
        public static final class Builder {

            private Image image;
            private List<Point> points;
            private List<Integer> labels;
            private boolean visualize;

            Builder(Image image) {
                this.image = image;
                points = new ArrayList<>();
                labels = new ArrayList<>();
            }

            /**
             * Adds a point to the {@code Sam2Input}.
             *
             * @param x the X coordinate
             * @param y the Y coordinate
             * @return the builder
             */
            public Builder addPoint(int x, int y) {
                return addPoint(x, y, 1);
            }

            /**
             * Adds a point to the {@code Sam2Input}.
             *
             * @param x the X coordinate
             * @param y the Y coordinate
             * @param label the label of the point, 0 for background, 1 for foreground
             * @return the builder
             */
            public Builder addPoint(int x, int y, int label) {
                return addPoint(new Point(x, y), label);
            }

            /**
             * Adds a point to the {@code Sam2Input}.
             *
             * @param point the point on image
             * @param label the label of the point, 0 for background, 1 for foreground
             * @return the builder
             */
            public Builder addPoint(Point point, int label) {
                points.add(point);
                labels.add(label);
                return this;
            }

            /**
             * Adds a box area to the {@code Sam2Input}.
             *
             * @param x the left coordinate
             * @param y the top coordinate
             * @param right the right coordinate
             * @param bottom the bottom coordinate
             * @return the builder
             */
            public Builder addBox(int x, int y, int right, int bottom) {
                addPoint(new Point(x, y), 2);
                addPoint(new Point(right, bottom), 3);
                return this;
            }

            /**
             * Sets the visualize for the {@code Sam2Input}.
             *
             * @return the builder
             */
            public Builder visualize() {
                visualize = true;
                return this;
            }

            /**
             * Builds the {@code Sam2Input}.
             *
             * @return the new {@code Sam2Input}
             */
            public Sam2Input build() {
                Point[] location = points.toArray(new Point[0]);
                int[] array = labels.stream().mapToInt(Integer::intValue).toArray();
                return new Sam2Input(image, location, array, visualize);
            }
        }

        private static final class Location {
            String type;
            int[] data;
            int label;

            public void setType(String type) {
                this.type = type;
            }

            public void setData(int[] data) {
                this.data = data;
            }

            public void setLabel(int label) {
                this.label = label;
            }
        }

        private static final class Prompt {

            @SerializedName("image_url")
            String image;

            Location[] prompt;
            boolean visualize;

            public void setImage(String image) {
                this.image = image;
            }

            public void setPrompt(Location[] prompt) {
                this.prompt = prompt;
            }

            public void setVisualize(boolean visualize) {
                this.visualize = visualize;
            }
        }
    }
}
