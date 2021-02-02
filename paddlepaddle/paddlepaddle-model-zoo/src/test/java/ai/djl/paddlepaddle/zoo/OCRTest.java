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

package ai.djl.paddlepaddle.zoo;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.stream.IntStream;
import org.testng.annotations.Test;

public class OCRTest {

    @Test
    public void testOCR() throws IOException, ModelException, TranslateException {
        String url = "https://resources.djl.ai/images/flight_ticket.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        List<BoundingBox> boxes = detectWords(img);

        // load Character model
        Predictor<Image, String> recognizer = getRecognizer();
        Predictor<Image, Image> rotator = getRotateClassifer();
        for (int i = 0; i < boxes.size(); i++) {
            Image subImg = getSubImage(img, boxes.get(i));
            subImg = rotator.predict(subImg);
            String name = recognizer.predict(subImg);
            System.out.println(name);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<BoundingBox> detectWords(Image img)
            throws ModelException, IOException, TranslateException {
        Class<List<BoundingBox>> clazz = (Class) List.class;
        Criteria<Image, List<BoundingBox>> criteria =
                Criteria.builder()
                        .optEngine("PaddlePaddle")
                        .setTypes(Image.class, clazz)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/paddleOCR/mobile/det_db.zip")
                        .optTranslator(new WordsTranslator(960))
                        .build();

        try (ZooModel<Image, List<BoundingBox>> model = ModelZoo.loadModel(criteria);
                Predictor<Image, List<BoundingBox>> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    private static Predictor<Image, String> getRecognizer()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, String> criteria =
                Criteria.builder()
                        .optEngine("PaddlePaddle")
                        .setTypes(Image.class, String.class)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/paddleOCR/mobile/rec_crnn.zip")
                        .optTranslator(new CharacterRecognitionTranslator())
                        .build();

        ZooModel<Image, String> model = ModelZoo.loadModel(criteria);
        return model.newPredictor();
    }

    private static Predictor<Image, Image> getRotateClassifer()
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria<Image, Image> criteria =
                Criteria.builder()
                        .optEngine("PaddlePaddle")
                        .setTypes(Image.class, Image.class)
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/paddleOCR/mobile/cls.zip")
                        .optTranslator(new RotateTranslator())
                        .build();

        ZooModel<Image, Image> model = ModelZoo.loadModel(criteria);
        return model.newPredictor();
    }

    private static Image getSubImage(Image img, BoundingBox box) {
        Rectangle rect = box.getBounds();
        double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
        int width = img.getWidth();
        int height = img.getHeight();
        int[] recovered = {
            (int) (extended[0] * width),
            (int) (extended[1] * height),
            (int) (extended[2] * width),
            (int) (extended[3] * height)
        };
        return img.getSubimage(recovered[0], recovered[1], recovered[2], recovered[3]);
    }

    private static double[] extendRect(double xmin, double ymin, double width, double height) {
        double centerx = xmin + width / 2;
        double centery = ymin + height / 2;
        if (width > height) {
            width += height * 2.0;
            height *= 3.0;
        } else {
            height += width * 2.0;
            width *= 3.0;
        }
        double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
        double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
        double newWidth = newX + width > 1 ? 1 - newX : width;
        double newHeight = newY + height > 1 ? 1 - newY : height;
        return new double[] {newX, newY, newWidth, newHeight};
    }

    private static int[] scale(int h, int w, int max) {
        int localMax = Math.max(h, w);
        float scale = 1.0f;
        if (max < localMax) {
            scale = max * 1.0f / localMax;
        }
        // paddle model only take 32-based size
        return resize32(h * scale, w * scale);
    }

    private static int[] resize32(double h, double w) {
        double min = Math.min(h, w);
        if (min < 32) {
            h = 32.0 / min * h;
            w = 32.0 / min * w;
        }
        int h32 = (int) h / 32;
        int w32 = (int) w / 32;
        return new int[] {h32 * 32, w32 * 32};
    }

    static class WordsTranslator implements Translator<Image, List<BoundingBox>> {

        private int maxLength;

        public WordsTranslator(int maxLength) {
            this.maxLength = maxLength;
        }

        @Override
        public List<BoundingBox> processOutput(TranslatorContext ctx, NDList list)
                throws IOException {
            NDArray result = list.singletonOrThrow();
            result = result.squeeze().mul(255f).toType(DataType.UINT8, true).neq(0);
            boolean[] flattened = result.toBooleanArray();
            Shape shape = result.getShape();
            int w = (int) shape.get(0);
            int h = (int) shape.get(1);
            boolean[][] grid = new boolean[w][h];
            IntStream.range(0, flattened.length)
                    .parallel()
                    .forEach(i -> grid[i / h][i % h] = flattened[i]);
            return new BoundFinder(grid).getBoxes();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray img = input.toNDArray(ctx.getNDManager());
            int h = input.getHeight();
            int w = input.getWidth();
            int[] hw = scale(h, w, maxLength);

            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img);
            img =
                    NDImageUtils.normalize(
                            img,
                            new float[] {0.485f, 0.456f, 0.406f},
                            new float[] {0.229f, 0.224f, 0.225f});
            img = img.expandDims(0);
            return new NDList(img);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }

    static class CharacterRecognitionTranslator implements Translator<Image, String> {

        private List<String> table;

        /** {@inheritDoc} */
        @Override
        public void prepare(NDManager manager, Model model) throws IOException {
            try (InputStream is = model.getArtifact("rec_crnn/ppocr_keys_v1.txt").openStream()) {
                table = Utils.readLines(is, true);
                table.add(0, "blank");
                table.add("");
            }
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) throws IOException {
            StringBuilder sb = new StringBuilder();
            NDArray tokens = list.singletonOrThrow();
            long[] indices = tokens.get(0).argMax(1).toLongArray();
            int lastIdx = 0;
            for (int i = 0; i < indices.length; i++) {
                if (indices[i] > 0 && !(i > 0 && indices[i] == lastIdx)) {
                    sb.append(table.get((int) indices[i]));
                }
            }
            return sb.toString();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray img = input.toNDArray(ctx.getNDManager());
            int[] hw = resize32(input.getHeight(), input.getWidth());
            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img).sub(0.5f).div(0.5f);
            img = img.expandDims(0);
            return new NDList(img);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }

    static class RotateTranslator implements Translator<Image, Image> {

        @Override
        public Image processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDArray img = (NDArray) ctx.getAttachment("img");
            float[] prob = list.singletonOrThrow().toFloatArray();
            if (prob[1] > prob[0] && prob[1] > 0.8f) {
                img = NDImageUtils.rotate90(img, 1);
            }
            return ImageFactory.getInstance().fromNDArray(img);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray img = input.toNDArray(ctx.getNDManager());
            if (input.getHeight() * 1.0 / input.getWidth() > 1.5) {
                img = NDImageUtils.rotate90(img, 1);
            }
            ctx.setAttachment("img", img);
            int[] hw = resize32(input.getHeight(), input.getWidth());
            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img).sub(0.5f).div(0.5f);
            img = img.expandDims(0);
            return new NDList(img);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
