/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.BaseImageTranslator.BaseBuilder;
import ai.djl.modality.nlp.NlpUtils;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.TextCleaner;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;
import ai.djl.util.Utils;

import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** A translator for Yolo-world models. */
public class YoloWorldTranslator
        implements NoBatchifyTranslator<VisionLanguageInput, DetectedObjects> {

    private static final int MAX_DETECTION = 300;
    private static final int[] AXIS_0 = {0};

    private SimpleBpeTokenizer tokenizer;
    private BaseImageTranslator<?> imageProcessor;
    private Predictor<NDList, NDList> predictor;
    private String clipModelPath;
    private float threshold;
    private float nmsThreshold;

    YoloWorldTranslator(Builder builder) {
        this.imageProcessor = new BaseImagePreProcessor(builder);
        this.threshold = builder.threshold;
        this.nmsThreshold = builder.nmsThreshold;
        this.clipModelPath = builder.clipModelPath;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        Model model = ctx.getModel();
        Path modelPath = model.getModelPath();
        Path path = Paths.get(clipModelPath);
        if (!path.isAbsolute() && Files.notExists(path)) {
            path = modelPath.resolve(clipModelPath);
        }
        if (!Files.exists(path)) {
            throw new IOException("clip model not found: " + clipModelPath);
        }
        NDManager manager = ctx.getNDManager();
        Model clip = manager.getEngine().newModel("clip", manager.getDevice());
        clip.load(path);
        predictor = clip.newPredictor(new NoopTranslator(null));
        model.getNDManager().attachInternal(NDManager.nextUid(), predictor);
        model.getNDManager().attachInternal(NDManager.nextUid(), clip);
        tokenizer = SimpleBpeTokenizer.newInstance(modelPath);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, VisionLanguageInput input)
            throws TranslateException {
        NDManager manager = ctx.getNDManager();

        String[] candidates = input.getCandidates();
        if (candidates == null || candidates.length == 0) {
            throw new TranslateException("Missing candidates in input");
        }

        int[][] tokenIds = tokenizer.batchEncode(candidates);
        NDArray textFeature = predictor.predict(new NDList(manager.create(tokenIds))).get(0);

        Image img = input.getImage();
        NDList imageFeatures = imageProcessor.processInput(ctx, img);
        NDArray array = imageFeatures.get(0).expandDims(0);

        ctx.setAttachment("candidates", candidates);
        return new NDList(textFeature, array);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        List<String> classes = Arrays.asList((String[]) ctx.getAttachment("candidates"));
        int width = (Integer) ctx.getAttachment("width");
        int height = (Integer) ctx.getAttachment("height");

        NDArray pred = list.get(0);
        pred = pred.squeeze(0);
        int boxIndex = classes.size() + 4;

        NDArray candidates = pred.get("4:" + boxIndex).max(AXIS_0).gt(threshold);
        pred = pred.transpose();
        NDArray sub = pred.get("..., :4");
        sub = YoloTranslator.xywh2xyxy(sub);
        pred = sub.concat(pred.get("..., 4:"), -1);
        pred = pred.get(candidates);

        NDList split = pred.split(new long[] {4, boxIndex}, 1);
        NDArray box = split.get(0);

        int numBox = Math.toIntExact(box.getShape().get(0));

        float[] buf = box.toFloatArray();
        float[] confidences = split.get(1).toFloatArray();
        long[] ids = split.get(1).argMax(1).toLongArray();

        List<Rectangle> boxes = new ArrayList<>(numBox);
        List<Double> scores = new ArrayList<>(numBox);
        for (int i = 0; i < numBox; ++i) {
            float xPos = buf[i * 4];
            float yPos = buf[i * 4 + 1];
            float w = buf[i * 4 + 2] - xPos;
            float h = buf[i * 4 + 3] - yPos;
            Rectangle rect = new Rectangle(xPos, yPos, w, h);
            boxes.add(rect);
            scores.add((double) confidences[i]);
        }
        List<Integer> nms = Rectangle.nms(boxes, scores, nmsThreshold);
        if (nms.size() > MAX_DETECTION) {
            nms = nms.subList(0, MAX_DETECTION);
        }

        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();
        for (int index : nms) {
            int id = (int) ids[index];
            retClasses.add(classes.get(id));
            retProbs.add((double) confidences[id]);
            Rectangle rect = boxes.get(index);
            rect =
                    new Rectangle(
                            rect.getX() / width,
                            rect.getY() / height,
                            rect.getWidth() / width,
                            rect.getHeight() / height);
            retBB.add(rect);
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@link YoloWorldTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code YoloWorldTranslator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);
        return builder;
    }

    /** The builder for {@link YoloWorldTranslator}. */
    public static class Builder extends BaseBuilder<Builder> {

        float threshold = 0.25f;
        float nmsThreshold = 0.7f;
        String clipModelPath = "clip.pt";

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for prediction accuracy
         * @return the builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return self();
        }

        /**
         * Sets the NMS threshold.
         *
         * @param nmsThreshold the NMS threshold
         * @return this builder
         */
        public Builder optNmsThreshold(float nmsThreshold) {
            this.nmsThreshold = nmsThreshold;
            return this;
        }

        /**
         * Sets the clip model file path, default value "clip.pt".
         *
         * @param clipModelPath the clip model file path
         * @return this builder
         */
        public Builder optClipModelPath(String clipModelPath) {
            this.clipModelPath = clipModelPath;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            optThreshold(ArgumentsUtil.floatValue(arguments, "threshold", threshold));
            optNmsThreshold(ArgumentsUtil.floatValue(arguments, "nmsThreshold", nmsThreshold));
            optClipModelPath(ArgumentsUtil.stringValue(arguments, "clipModelPath", "clip.pt"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public YoloWorldTranslator build() {
            return new YoloWorldTranslator(this);
        }
    }

    static final class SimpleBpeTokenizer {

        private static final int MIN_CONTEXT_LENGTH = 77;
        private static final int MAX_CONTEXT_LENGTH = 512;
        private static final Type MAP_TYPE = new TypeToken<Map<String, Integer>>() {}.getType();

        private Map<String, Integer> vocabulary;
        private Map<Pair<String, String>, Integer> ranks;
        private int sot;
        private int eot;

        SimpleBpeTokenizer(
                Map<String, Integer> vocabulary, Map<Pair<String, String>, Integer> ranks) {
            this.vocabulary = vocabulary;
            this.ranks = ranks;
            sot = vocabulary.get("<|startoftext|>");
            eot = vocabulary.get("<|endoftext|>");
        }

        static SimpleBpeTokenizer newInstance(Path modelPath) throws IOException {
            Path vocab = modelPath.resolve("vocab.json");
            Path merges = modelPath.resolve("merges.txt");
            Map<Pair<String, String>, Integer> ranks = new ConcurrentHashMap<>();
            List<String> lines = Utils.readLines(merges);
            lines = lines.subList(1, lines.size());
            int index = 0;
            for (String line : lines) {
                String[] tok = line.split(" ");
                ranks.put(new Pair<>(tok[0], tok[1]), index++);
            }

            try (Reader reader = Files.newBufferedReader(vocab)) {
                Map<String, Integer> vocabulary = JsonUtils.GSON.fromJson(reader, MAP_TYPE);
                return new SimpleBpeTokenizer(vocabulary, ranks);
            }
        }

        int[][] batchEncode(String[] inputs) {
            List<List<Integer>> list = new ArrayList<>();
            int contextLength = 0;
            for (String input : inputs) {
                List<Integer> ids = encode(input);
                int size = ids.size();
                if (size > MAX_CONTEXT_LENGTH) {
                    ids = ids.subList(0, MAX_CONTEXT_LENGTH);
                }
                contextLength = Math.max(contextLength, size);

                list.add(ids);
            }
            contextLength = Math.max(contextLength, MIN_CONTEXT_LENGTH);

            int[][] tokenIds = new int[inputs.length][contextLength];
            int row = 0;
            for (List<Integer> ids : list) {
                for (int col = 0; col < ids.size(); ++col) {
                    tokenIds[row][col] = ids.get(col);
                }
                ++row;
            }
            return tokenIds;
        }

        List<Integer> encode(String text) {
            List<String> tokens = new ArrayList<>(Collections.singletonList(text));
            List<TextProcessor> processors = new ArrayList<>();
            processors.add(new LowerCaseConvertor());
            processors.add(new TextCleaner(NlpUtils::isWhiteSpace, ' '));
            processors.add(new PunctuationSeparator());
            for (TextProcessor processor : processors) {
                tokens = processor.preprocess(tokens);
            }
            List<Integer> idx = new ArrayList<>();
            idx.add(sot);
            for (String token : tokens) {
                String bpe = bpe(token);
                idx.add(vocabulary.get(bpe));
            }
            idx.add(eot);
            return idx;
        }

        private String bpe(String token) {
            char[] chars = token.toCharArray();
            List<String> word = new ArrayList<>(chars.length);
            for (char c : chars) {
                word.add(String.valueOf(c));
            }
            word.set(word.size() - 1, word.get(word.size() - 1) + "</w>");
            Set<Pair<String, String>> pairs = getPairs(word);
            if (pairs.isEmpty()) {
                return token + "</w>";
            }

            while (true) {
                Pair<String, String> min =
                        Collections.min(
                                pairs,
                                (o1, o2) ->
                                        Integer.compare(
                                                ranks.getOrDefault(o1, Integer.MAX_VALUE),
                                                ranks.getOrDefault(o2, Integer.MAX_VALUE)));
                if (!ranks.containsKey(min)) {
                    break;
                }
                List<String> newWord = new ArrayList<>();
                String first = min.getKey();
                String second = min.getValue();
                int i = 0;
                while (i < word.size()) {
                    List<String> subList = word.subList(i, word.size());
                    int j = subList.indexOf(first);
                    if (j < 0) {
                        newWord.addAll(word.subList(i, word.size()));
                        break;
                    } else {
                        j += i;
                    }
                    newWord.addAll(word.subList(i, j));
                    i = j;

                    if (word.get(i).equals(first)
                            && i < word.size() - 1
                            && word.get(i + 1).equals(second)) {
                        newWord.add(first + second);
                        i += 2;
                    } else {
                        newWord.add(word.get(i));
                        i++;
                    }
                }

                word = newWord;
                if (word.size() == 1) {
                    break;
                } else {
                    pairs = getPairs(word);
                }
            }
            return String.join(" ", word);
        }

        private Set<Pair<String, String>> getPairs(List<String> word) {
            if (word.size() < 2) {
                return Collections.emptySet();
            }

            Set<Pair<String, String>> pairs = new HashSet<>();
            String prev = word.get(0);
            for (int i = 1; i < word.size(); ++i) {
                pairs.add(new Pair<>(prev, word.get(i)));
                prev = word.get(i);
            }
            return pairs;
        }
    }
}
