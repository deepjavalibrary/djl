/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.examples.training.util.Arguments;
import ai.djl.modality.nlp.preprocess.UnicodeNormalizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.transformer.BertBlock;
import ai.djl.nn.transformer.BertPretrainingBlock;
import ai.djl.nn.transformer.BertPretrainingLoss;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.TruncatedNormalInitializer;
import ai.djl.training.listener.TrainingListener.Defaults;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.PolynomialDecayTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.tracker.WarmUpTracker;
import ai.djl.training.tracker.WarmUpTracker.Mode;
import ai.djl.translate.Batchifier;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** Simple example that performs Bert pretraining on the java source files in this repo. */
@SuppressWarnings("PMD")
public final class TrainBertOnCode {

    private static final String UNK = "<unk>";
    private static final String CLS = "<cls>";
    private static final String SEP = "<sep>";
    private static final String MSK = "<msk>";

    private static final List<String> RESERVED_TOKENS =
            Collections.unmodifiableList(Arrays.asList(UNK, CLS, SEP, MSK));

    private static final int UNK_ID = RESERVED_TOKENS.indexOf(UNK);
    private static final int CLS_ID = RESERVED_TOKENS.indexOf(CLS);
    private static final int SEP_ID = RESERVED_TOKENS.indexOf(SEP);
    private static final int MSK_ID = RESERVED_TOKENS.indexOf(MSK);

    private static final int MAX_SEQUENCE_LENGTH = 128;
    private static final int MAX_MASKING_PER_INSTANCE = 20;
    private static final int BATCH_SIZE = 48;
    private static final int EPOCHS = 10;
    private static final BertBlock.Builder BERT_BUILDER = BertBlock.builder().micro();

    private TrainBertOnCode() {}

    public static void main(String[] args) {
        TrainBertOnCode.runExample(args);
    }

    public static TrainingResult runExample(String[] args) {
        BertArguments arguments = (BertArguments) new BertArguments().parseArgs(args);
        Random rand = new Random(89724308);
        // get all applicable files
        List<Path> files = listSourceFiles(new File(".").toPath());
        // read & tokenize them
        List<ParsedFile> parsedFiles =
                files.stream().map(TrainBertOnCode::parseFile).collect(Collectors.toList());
        // determine dictionary
        Map<String, Long> countedTokens = countTokens(parsedFiles);
        Dictionary dictionary = buildDictionary(countedTokens, 35000);

        // Create model & trainer
        try (Model model = createBertPretrainingModel(dictionary)) {

            try (Trainer trainer = createBertPretrainingTrainer(model, arguments)) {

                // Initialize training
                Shape inputShape = new Shape(MAX_SEQUENCE_LENGTH, 512);
                trainer.initialize(inputShape, inputShape, inputShape, inputShape);

                trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
                for (int epoch = 0; epoch < arguments.getEpoch(); ++epoch) {
                    List<MaskedInstance> maskedInstances =
                            createEpochData(rand, dictionary, parsedFiles, arguments);
                    for (int idx = BATCH_SIZE; idx < maskedInstances.size(); ++idx) {
                        try (NDManager ndManager = trainer.getManager().newSubManager()) {
                            List<MaskedInstance> batchData =
                                    maskedInstances.subList(idx - BATCH_SIZE, idx);
                            Batch batch =
                                    createBatch(ndManager, batchData, idx, maskedInstances.size());
                            EasyTrain.trainBatch(trainer, batch);
                        }
                    }
                    trainer.notifyListeners(listener -> listener.onEpoch(trainer));
                }
                trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));
                return trainer.getTrainingResult();
            }
        }
    }

    private static Model createBertPretrainingModel(Dictionary dictionary) {
        Model model = Model.newInstance("Bert Pretraining");
        model.setBlock(
                new BertPretrainingBlock(
                        BERT_BUILDER.setTokenDictionarySize(dictionary.tokens.size())));
        model.getBlock()
                .setInitializer(new TruncatedNormalInitializer(0.02f), Parameter.Type.WEIGHT);
        return model;
    }

    private static Trainer createBertPretrainingTrainer(Model model, BertArguments arguments) {
        Tracker learningRateTracker =
                WarmUpTracker.builder()
                        .optWarmUpBeginValue(0f)
                        .optWarmUpSteps(1000)
                        .optWarmUpMode(Mode.LINEAR)
                        .setMainTracker(
                                PolynomialDecayTracker.builder()
                                        .setBaseValue(5e-5f)
                                        .setEndLearningRate(5e-5f / 1000)
                                        .setDecaySteps(100000)
                                        .optPower(1f)
                                        .build())
                        .build();
        Optimizer optimizer =
                Adam.builder()
                        .optEpsilon(1e-5f)
                        .optLearningRateTracker(learningRateTracker)
                        .build();
        TrainingConfig trainingConfig =
                new DefaultTrainingConfig(new BertPretrainingLoss())
                        .optOptimizer(optimizer)
                        .optDevices(Device.getDevices(arguments.getMaxGpus()))
                        .optExecutorService(ForkJoinPool.commonPool())
                        .addTrainingListeners(Defaults.logging());
        return model.newTrainer(trainingConfig);
    }

    private static List<MaskedInstance> createEpochData(
            Random rand,
            Dictionary dictionary,
            List<ParsedFile> parsedFiles,
            BertArguments arguments) {
        // turn data into sentence pairs containing consecutive lines
        List<SentencePair> sentencePairs = new ArrayList<>();
        parsedFiles.forEach(parsedFile -> parsedFile.addToSentencePairs(sentencePairs));
        Collections.shuffle(sentencePairs, rand);
        // swap sentences with 50% probability for next sentence task
        for (int idx = 1; idx < sentencePairs.size(); idx += 2) {
            sentencePairs.get(idx - 1).maybeSwap(rand, sentencePairs.get(idx));
        }
        // Create masked instances for training
        return sentencePairs
                .stream()
                .limit(arguments.getLimit())
                .map(
                        sentencePair ->
                                new MaskedInstance(
                                        rand,
                                        dictionary,
                                        sentencePair,
                                        MAX_SEQUENCE_LENGTH,
                                        MAX_MASKING_PER_INSTANCE))
                .collect(Collectors.toList());
    }

    private static Batch createBatch(
            NDManager ndManager, List<MaskedInstance> instances, int idx, int dataSize) {
        NDList inputs =
                new NDList(
                        batchFromList(ndManager, instances, MaskedInstance::getTokenIds),
                        batchFromList(ndManager, instances, MaskedInstance::getTypeIds),
                        batchFromList(ndManager, instances, MaskedInstance::getInputMask),
                        batchFromList(ndManager, instances, MaskedInstance::getMaskedPositions));
        NDList labels =
                new NDList(
                        nextSentenceLabelsFromList(ndManager, instances),
                        batchFromList(ndManager, instances, MaskedInstance::getMaskedIds),
                        batchFromList(ndManager, instances, MaskedInstance::getLabelMask));
        return new Batch(
                ndManager,
                inputs,
                labels,
                instances.size(),
                Batchifier.STACK,
                Batchifier.STACK,
                idx,
                dataSize);
    }

    private static NDArray batchFromList(NDManager ndManager, List<int[]> batchData) {
        int[][] arrays = new int[batchData.size()][];
        for (int idx = 0; idx < batchData.size(); ++idx) {
            arrays[idx] = batchData.get(idx);
        }
        return ndManager.create(arrays);
    }

    private static NDArray batchFromList(
            NDManager ndManager,
            List<MaskedInstance> instances,
            Function<MaskedInstance, int[]> f) {
        return batchFromList(ndManager, instances.stream().map(f).collect(Collectors.toList()));
    }

    private static NDArray nextSentenceLabelsFromList(
            NDManager ndManager, List<MaskedInstance> instances) {
        int[] nextSentenceLabels = new int[instances.size()];
        for (int idx = 0; idx < nextSentenceLabels.length; ++idx) {
            nextSentenceLabels[idx] = instances.get(idx).getNextSentenceLabel();
        }
        return ndManager.create(nextSentenceLabels);
    }

    private static List<Path> listSourceFiles(Path root) {
        try {
            return Files.walk(root)
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().toLowerCase().endsWith(".java"))
                    .collect(Collectors.toList());
        } catch (IOException ioe) {
            throw new RuntimeException("Could not list files for " + root);
        }
    }

    private static String normalizeLine(String line) {
        if (line.isEmpty()) {
            return line;
        }
        // in source code, preceding whitespace is relevant, trailing ws is not
        // so we get the index of the last non ws char
        String unicodeNormalized = UnicodeNormalizer.normalizeDefault(line);
        int endIdx = line.length() - 1;
        while (endIdx >= 0 && Character.isWhitespace(unicodeNormalized.charAt(endIdx))) {
            endIdx--;
        }
        return line.substring(0, endIdx + 1);
    }

    private static List<String> fileToLines(Path file) {
        try {
            return Files.lines(file, StandardCharsets.UTF_8)
                    .map(TrainBertOnCode::normalizeLine)
                    .filter(line -> !line.trim().isEmpty())
                    .collect(Collectors.toList());
        } catch (IOException ioe) {
            throw new RuntimeException("Could not read file " + file, ioe);
        }
    }

    private static List<String> tokenizeLine(String normalizedLine) {
        // note: we work on chars, as this is a quick'n'dirty example - in the real world,
        // we should work on codepoints.
        if (normalizedLine.isEmpty()) {
            return Collections.emptyList();
        }
        if (normalizedLine.length() == 1) {
            return Collections.singletonList(normalizedLine);
        }
        List<String> result = new ArrayList<>();
        final int length = normalizedLine.length();
        final StringBuilder currentToken = new StringBuilder();
        for (int idx = 0; idx <= length; ++idx) {
            char c = idx < length ? normalizedLine.charAt(idx) : 0;
            boolean isAlphabetic = Character.isAlphabetic(c);
            boolean isUpperCase = Character.isUpperCase(c);
            if (c == 0 || !isAlphabetic || isUpperCase) {
                // we have reached the end of the string, encountered something other than a letter
                // or reached a new part of a camel-cased word - emit a new token
                if (currentToken.length() > 0) {
                    result.add(currentToken.toString().toLowerCase());
                    currentToken.setLength(0);
                }
                // if we haven't reached the end, we need to use the char
                if (c != 0) {
                    if (!isAlphabetic) {
                        // the char is not alphabetic, turn it into a separate token
                        result.add(Character.toString(c));
                    } else {
                        currentToken.append(c);
                    }
                }
            } else {
                // we have a new char to append to the current token
                currentToken.append(c);
            }
        }
        return result;
    }

    private static Map<String, Long> countTokens(List<ParsedFile> parsedFiles) {
        Map<String, Long> result = new HashMap<>(50000);
        parsedFiles.forEach(parsedFile -> countTokens(parsedFile, result));
        return result;
    }

    private static void countTokens(ParsedFile parsedFile, Map<String, Long> result) {
        parsedFile.tokenizedLines.forEach(tokens -> countTokens(tokens, result));
    }

    private static void countTokens(List<String> tokenizedLine, Map<String, Long> result) {
        for (String token : tokenizedLine) {
            long count = result.getOrDefault(token, 0L);
            result.put(token, count + 1);
        }
    }

    private static ParsedFile parseFile(Path file) {
        List<String> normalizedLines =
                fileToLines(file)
                        .stream()
                        .map(TrainBertOnCode::normalizeLine)
                        .filter(line -> !line.isEmpty())
                        .collect(Collectors.toList());
        List<List<String>> tokens =
                normalizedLines
                        .stream()
                        .map(TrainBertOnCode::tokenizeLine)
                        .collect(Collectors.toList());
        return new ParsedFile(file, normalizedLines, tokens);
    }

    private static final Dictionary buildDictionary(Map<String, Long> countedTokens, int maxSize) {
        if (maxSize < RESERVED_TOKENS.size()) {
            throw new IllegalArgumentException(
                    "Dictionary needs at least size "
                            + RESERVED_TOKENS.size()
                            + " to account for reserved tokens.");
        }
        ArrayList<String> result = new ArrayList<>(maxSize);
        result.addAll(RESERVED_TOKENS);
        List<String> sortedByFrequency =
                countedTokens
                        .entrySet()
                        .stream()
                        .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());
        int idx = 0;
        while (result.size() < maxSize && idx < sortedByFrequency.size()) {
            result.add(sortedByFrequency.get(idx));
            idx++;
        }
        return new Dictionary(result);
    }

    private static final class ParsedFile {
        final Path sourceFile;
        final List<String> normalizedLines;
        final List<List<String>> tokenizedLines;

        private ParsedFile(
                Path sourceFile, List<String> normalizedLines, List<List<String>> tokenizedLines) {
            this.sourceFile = sourceFile;
            this.normalizedLines = normalizedLines;
            this.tokenizedLines = tokenizedLines;
        }

        public String toDebugString() {
            return sourceFile
                    + "\n"
                    + "=======================\n"
                    + tokenizedLines
                            .stream()
                            .map(tokens -> String.join("|", tokens))
                            .collect(Collectors.joining("\n"));
        }

        public void addToSentencePairs(List<SentencePair> sentencePairs) {
            for (int idx = 1; idx < tokenizedLines.size(); idx += 2) {
                sentencePairs.add(
                        new SentencePair(
                                new ArrayList<>(tokenizedLines.get(idx - 1)),
                                new ArrayList<>(tokenizedLines.get(idx))));
            }
        }
    }

    /** Helper class to preprocess data for the next sentence prediction task. */
    private static final class SentencePair {
        ArrayList<String> sentenceA;
        ArrayList<String> sentenceB;
        boolean consecutive = true;

        private SentencePair(ArrayList<String> sentenceA, ArrayList<String> sentenceB) {
            this.sentenceA = sentenceA;
            this.sentenceB = sentenceB;
        }

        public void maybeSwap(Random rand, SentencePair other) {
            if (rand.nextBoolean()) {
                ArrayList<String> otherA = other.sentenceA;
                other.sentenceA = this.sentenceA;
                this.sentenceA = otherA;
                this.consecutive = false;
                other.consecutive = false;
            }
        }

        public int getTotalLength() {
            return sentenceA.size() + sentenceB.size();
        }

        public void truncateToTotalLength(int totalLength) {
            int count = 0;
            while (getTotalLength() > totalLength) {
                if (count % 2 == 0 && !sentenceA.isEmpty()) {
                    sentenceA.remove(sentenceA.size() - 1);
                } else if (!sentenceB.isEmpty()) {
                    sentenceB.remove(sentenceB.size() - 1);
                }
                count++;
            }
        }
    }

    /** A single bert pretraining instance. Applies masking to a given sentence pair. */
    private static final class MaskedInstance {
        final Dictionary dictionary;
        final SentencePair originalSentencePair;
        final ArrayList<String> label;
        final ArrayList<String> masked;
        final int seperatorIdx;
        final ArrayList<Integer> typeIds;
        final ArrayList<Integer> maskedIndices;
        final int maxSequenceLength;
        final int maxMasking;

        public MaskedInstance(
                Random rand,
                Dictionary dictionary,
                SentencePair originalSentencePair,
                int maxSequenceLength,
                int maxMasking) {
            this.dictionary = dictionary;
            this.originalSentencePair = originalSentencePair;
            this.maxSequenceLength = maxSequenceLength;
            this.maxMasking = maxMasking;
            // Create input sequence of right length with control tokens added;
            // account cls & sep tokens
            int maxTokenCount = maxSequenceLength - 3;
            originalSentencePair.truncateToTotalLength(maxTokenCount);
            label = new ArrayList<>(originalSentencePair.getTotalLength() + 3);
            label.add(CLS);
            label.addAll(originalSentencePair.sentenceA);
            seperatorIdx = label.size();
            label.add(SEP);
            label.addAll(originalSentencePair.sentenceB);
            label.add(SEP);
            masked = new ArrayList<>(label);
            // create type tokens (0 = sentence a, 1, sentence b)
            typeIds = new ArrayList<>(label.size());
            int typeId = 0;
            for (int idx = 0; idx < label.size(); ++idx) {
                typeIds.add(typeId);
                if (label.get(idx) == SEP) {
                    typeId++;
                }
            }
            // Randomly pick 20% of indices to mask
            int maskedCount = Math.min((int) (0.2f * label.size()), maxMasking);
            List<Integer> temp =
                    IntStream.range(0, label.size()).boxed().collect(Collectors.toList());
            Collections.shuffle(temp, rand);
            maskedIndices = new ArrayList<>(temp.subList(0, maskedCount));
            Collections.sort(maskedIndices);
            // Perform masking of these indices
            for (int maskedIdx : maskedIndices) {
                // decide what to mask
                float r = rand.nextFloat();
                if (r < 0.8f) { // 80% probability -> mask
                    masked.set(maskedIdx, MSK);
                } else if (r < 0.9f) { // 10% probability -> random token
                    masked.set(maskedIdx, dictionary.getRandomToken(rand));
                } // 10% probability: leave token as-is
            }
        }

        public int[] getTokenIds() {
            int[] result = new int[maxSequenceLength];
            for (int idx = 0; idx < masked.size(); ++idx) {
                result[idx] = dictionary.getId(masked.get(idx));
            }
            return result;
        }

        public int[] getTypeIds() {
            int[] result = new int[maxSequenceLength];
            for (int idx = 0; idx < typeIds.size(); ++idx) {
                result[idx] = typeIds.get(idx);
            }
            return result;
        }

        public int[] getInputMask() {
            int[] result = new int[maxSequenceLength];
            for (int idx = 0; idx < typeIds.size(); ++idx) {
                result[idx] = 1;
            }
            return result;
        }

        public int[] getMaskedPositions() {
            int[] result = new int[maxMasking];
            for (int idx = 0; idx < maskedIndices.size(); ++idx) {
                result[idx] = maskedIndices.get(idx);
            }
            return result;
        }

        public int getNextSentenceLabel() {
            return originalSentencePair.consecutive ? 1 : 0;
        }

        public int[] getMaskedIds() {
            int[] result = new int[maxMasking];
            for (int idx = 0; idx < maskedIndices.size(); ++idx) {
                result[idx] = dictionary.getId(label.get(maskedIndices.get(idx)));
            }
            return result;
        }

        public int[] getLabelMask() {
            int[] result = new int[maxMasking];
            for (int idx = 0; idx < maskedIndices.size(); ++idx) {
                result[idx] = 1;
            }
            return result;
        }

        public String toDebugString() {
            return originalSentencePair.consecutive
                    + "\n"
                    + String.join("", label)
                    + "\n"
                    + String.join("", masked);
        }
    }

    /** Helper class to create a token to id mapping. */
    private static final class Dictionary {
        final ArrayList<String> tokens;
        final Map<String, Integer> tokenToId;

        private Dictionary(ArrayList<String> tokens) {
            this.tokens = tokens;
            this.tokenToId = new HashMap<>(tokens.size());
            for (int idx = 0; idx < tokens.size(); ++idx) {
                tokenToId.put(tokens.get(idx), idx);
            }
        }

        public String getToken(int id) {
            return id >= 0 && id < tokens.size() ? tokens.get(id) : UNK;
        }

        public int getId(String token) {
            return tokenToId.getOrDefault(token, UNK_ID);
        }

        public String toDebugString() {
            return String.join("\n", tokens);
        }

        public List<Integer> toIds(final List<String> tokens) {
            return tokens.stream().map(this::getId).collect(Collectors.toList());
        }

        public List<String> toTokens(final List<Integer> ids) {
            return ids.stream().map(this::getToken).collect(Collectors.toList());
        }

        public String getRandomToken(Random rand) {
            return tokens.get(rand.nextInt(tokens.size()));
        }
    }

    private static class BertArguments extends Arguments {

        @Override
        protected void initialize() {
            super.initialize();
            epoch = EPOCHS;
            batchSize = BATCH_SIZE;
        }
    }
}
