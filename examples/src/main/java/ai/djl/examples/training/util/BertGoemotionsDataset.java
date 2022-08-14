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
package ai.djl.examples.training.util;

import ai.djl.basicdataset.nlp.GoEmotions;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** An example of transform goemotions datasetset into which could be used by Bert model. */
public class BertGoemotionsDataset implements Dataset {
    private static final String CLS = "<cls>";
    private static final String SEP = "<sep>";
    private static final String MSK = "<msk>";

    private static final int MAX_SEQUENCE_LENGTH = 128;
    private static final int MAX_MASKING_PER_INSTANCE = 10;

    private ParsedFile parsedFiles;
    private Vocabulary vocabulary;
    private Random rand;
    private int batchSize;
    private long epochLimit;
    private NDManager manager;

    private GoEmotions goEmotions;

    public BertGoemotionsDataset(int batchSize, long epochLimit, GoEmotions goEmotions) {
        this.batchSize = batchSize;
        this.epochLimit = epochLimit;
        this.manager = NDManager.newBaseManager();
        this.goEmotions = goEmotions;
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager) {
        return new EpochIterable();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {
        rand = new Random(89724308);
        // read & tokenize them
        parsedFiles = ParsedFile.parse(this.goEmotions);
        // determine vocabulary
        vocabulary = goEmotions.getVocabulary(true);
    }

    public long getVocabularySize() {
        return vocabulary.size();
    }

    public int getMaxSequenceLength() {
        return MAX_SEQUENCE_LENGTH;
    }

    private final class EpochIterable implements Iterable<Batch>, Iterator<Batch> {

        List<MaskedInstance> maskedInstances;
        int idx;

        public EpochIterable() {
            // turn data into sentence pairs containing consecutive lines
            List<SentencePair> sentencePairs = new ArrayList<>();
            parsedFiles.addToSentencePairs(sentencePairs);
            Collections.shuffle(sentencePairs, rand);
            // swap sentences with 50% probability for next sentence task
            for (int i = 1; i < sentencePairs.size(); i += 2) {
                sentencePairs.get(i - 1).maybeSwap(rand, sentencePairs.get(i));
            }
            // Create masked instances for training
            maskedInstances =
                    sentencePairs.stream()
                            .limit(epochLimit)
                            .map(
                                    sentencePair ->
                                            new MaskedInstance(
                                                    rand,
                                                    vocabulary,
                                                    sentencePair,
                                                    MAX_SEQUENCE_LENGTH,
                                                    MAX_MASKING_PER_INSTANCE))
                            .collect(Collectors.toList());
            idx = batchSize;
        }

        /** {@inheritDoc} */
        @Override
        public Iterator<Batch> iterator() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return idx < maskedInstances.size();
        }

        /** {@inheritDoc} */
        @Override
        public Batch next() {
            NDManager ndManager = manager.newSubManager();
            List<MaskedInstance> instances = maskedInstances.subList(idx - batchSize, idx);
            idx++;
            NDList inputs =
                    new NDList(
                            batchFromList(ndManager, instances, MaskedInstance::getTokenIds),
                            batchFromList(ndManager, instances, MaskedInstance::getTypeIds),
                            batchFromList(ndManager, instances, MaskedInstance::getInputMask),
                            batchFromList(
                                    ndManager, instances, MaskedInstance::getMaskedPositions));
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
                    instances.size());
        }

        private NDArray batchFromList(NDManager ndManager, List<int[]> batchData) {
            return ndManager.create(batchData.toArray(new int[0][]));
        }

        private NDArray batchFromList(
                NDManager ndManager,
                List<MaskedInstance> instances,
                Function<MaskedInstance, int[]> f) {
            return batchFromList(ndManager, instances.stream().map(f).collect(Collectors.toList()));
        }

        private NDArray nextSentenceLabelsFromList(
                NDManager ndManager, List<MaskedInstance> instances) {
            return ndManager.create(
                    instances.stream().mapToInt(MaskedInstance::getNextSentenceLabel).toArray());
        }
    }

    private static final class ParsedFile {
        final List<String> normalizedLines;
        final List<List<String>> tokenizedLines;

        private ParsedFile(List<String> normalizedLines, List<List<String>> tokenizedLines) {
            this.normalizedLines = normalizedLines;
            this.tokenizedLines = tokenizedLines;
        }

        private static ParsedFile parse(GoEmotions goEmotions) {
            List<String> normalizedLines = new ArrayList<>();
            for (int i = 0; i < goEmotions.size(); i++) {
                normalizedLines.add(goEmotions.getRawText(i, true));
            }
            List<List<String>> tokens =
                    normalizedLines.stream()
                            .map(ParsedFile::tokenizeLine)
                            .collect(Collectors.toList());
            return new ParsedFile(normalizedLines, tokens);
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
                    // we have reached the end of the string, encountered something other than a
                    // letter
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
        List<String> sentenceA;
        List<String> sentenceB;
        boolean consecutive = true;

        private SentencePair(List<String> sentenceA, List<String> sentenceB) {
            this.sentenceA = sentenceA;
            this.sentenceB = sentenceB;
        }

        public void maybeSwap(Random rand, SentencePair other) {
            if (rand.nextBoolean()) {
                List<String> otherA = other.sentenceA;
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
        final Vocabulary vocabulary;
        final SentencePair originalSentencePair;
        final List<String> label;
        final List<String> masked;
        final int seperatorIdx;
        final List<Integer> typeIds;
        final List<Integer> maskedIndices;
        final int maxSequenceLength;
        final int maxMasking;

        public MaskedInstance(
                Random rand,
                Vocabulary vocabulary,
                SentencePair originalSentencePair,
                int maxSequenceLength,
                int maxMasking) {
            this.vocabulary = vocabulary;
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
            for (String s : label) {
                typeIds.add(typeId);
                if (SEP.equals(s)) {
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
                    masked.set(maskedIdx, getRandomToken(rand));
                } // 10% probability: leave token as-is
            }
        }

        private String getRandomToken(Random rand) {
            return vocabulary.getToken(rand.nextInt(Math.toIntExact(vocabulary.size())));
        }

        public int[] getTokenIds() {
            int[] result = new int[maxSequenceLength];
            for (int idx = 0; idx < masked.size(); ++idx) {
                result[idx] = Math.toIntExact(vocabulary.getIndex(masked.get(idx)));
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
                result[idx] =
                        Math.toIntExact(vocabulary.getIndex(label.get(maskedIndices.get(idx))));
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
}
