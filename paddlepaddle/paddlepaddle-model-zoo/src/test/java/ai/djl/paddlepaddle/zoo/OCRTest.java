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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.IntStream;
import org.testng.annotations.Test;

public class OCRTest {

    @Test
    public void testOCR() throws IOException, ModelException, TranslateException {
        String url =
                "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/imgs/11.jpg";
        Image img = ImageFactory.getInstance().fromUrl(url);
        List<BoundingBox> boxes = detectWords(img);

        // load Character model
        // Predictor<Image, String> recognizer = getRecognizer();
        for (int i = 0; i < boxes.size(); i++) {
            Image subImg = getSubImage(img, boxes.get(i));
            saveImage(subImg, Integer.toString(i));
            // String name = recognizer.predict(subImg);
        }
    }

    private static void saveImage(Image img, String prefix) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve(prefix + ".png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
    }

    @SuppressWarnings("unchecked")
    private static List<BoundingBox> detectWords(Image img)
            throws ModelException, IOException, TranslateException {
        Class<List<BoundingBox>> clazz = (Class) List.class;
        Criteria<Image, List<BoundingBox>> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .optEngine("PaddlePaddle")
                        .setTypes(Image.class, clazz)
                        .optModelUrls(
                                "https://alpha-djl-demos.s3.amazonaws.com/model/paddleOCR/mobile/det_db.zip")
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
                                "https://alpha-djl-demos.s3.amazonaws.com/model/paddleOCR/mobile/rec_crnn.zip")
                        .optTranslator(new CharacterRecognitionTranslator())
                        .build();

        ZooModel<Image, String> model = ModelZoo.loadModel(criteria);
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
            width += height * 1.9;
            height *= 2.9;
        } else {
            height += width * 1.9;
            width *= 2.9;
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
        int h32 = (int) (h * scale) / 32;
        int w32 = (int) (w * scale) / 32;
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

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            return null;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray img = input.toNDArray(ctx.getNDManager());
            int h = input.getHeight();
            int w = input.getWidth();
            int[] hw = scale(h, w, Math.max(h, w));
            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img).sub(-0.5f).div(0.5f);
            img = img.expandDims(0);
            return new NDList(img);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
