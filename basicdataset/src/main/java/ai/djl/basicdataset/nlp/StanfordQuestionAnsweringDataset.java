/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset.nlp;

import ai.djl.Application.NLP;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.RawDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;

import com.google.gson.reflect.TypeToken;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of
 * questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every
 * question is a segment of text, or span, from the corresponding reading passage, or the question
 * might be unanswerable.
 *
 * @see <a href="https://rajpurkar.github.io/SQuAD-explorer/">Dataset website</a>
 */
@SuppressWarnings("unchecked")
public class StanfordQuestionAnsweringDataset extends TextDataset implements RawDataset<Object> {

    private static final String VERSION = "2.0";
    private static final String ARTIFACT_ID = "stanford-question-answer";

    /**
     * Store the information of each question, so that when function {@code get()} is called, we can
     * find the question corresponding to the index.
     */
    private List<QuestionInfo> questionInfoList;

    /**
     * Creates a new instance of {@link StanfordQuestionAnsweringDataset}.
     *
     * @param builder the builder object to build from
     */
    protected StanfordQuestionAnsweringDataset(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Creates a new builder to build a {@link StanfordQuestionAnsweringDataset}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    private Path prepareUsagePath(Progress progress) throws IOException {
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Path root = mrl.getRepository().getResourceDirectory(artifact);

        Path usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = Paths.get("train-v2.0.json");
                break;
            case TEST:
                usagePath = Paths.get("dev-v2.0.json");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        return root.resolve(usagePath);
    }

    /**
     * Prepares the dataset for use with tracked progress. In this method the JSON file will be
     * parsed. The question, context, title will be added to {@code sourceTextData} and the answers
     * will be added to {@code targetTextData}. Both of them will then be preprocessed.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     * @throws EmbeddingException if there are exceptions during the embedding process
     */
    @Override
    public void prepare(Progress progress) throws IOException, EmbeddingException {
        if (prepared) {
            return;
        }
        Path usagePath = prepareUsagePath(progress);

        Map<String, Object> data;
        try (BufferedReader reader = Files.newBufferedReader(usagePath)) {
            data =
                    JsonUtils.GSON_PRETTY.fromJson(
                            reader, new TypeToken<Map<String, Object>>() {}.getType());
        }
        List<Map<String, Object>> articles = (List<Map<String, Object>>) data.get("data");

        questionInfoList = new ArrayList<>();
        List<String> sourceTextData = new ArrayList<>();
        List<String> targetTextData = new ArrayList<>();

        // a nested loop to handle the nested json object
        List<Map<String, Object>> paragraphs;
        List<Map<String, Object>> questions;
        List<Map<String, Object>> answers;

        int titleIndex;
        int contextIndex;
        int questionIndex;
        int answerIndex;
        QuestionInfo questionInfo;
        for (Map<String, Object> article : articles) {
            titleIndex = sourceTextData.size();
            sourceTextData.add(article.get("title").toString());

            // iterate through the paragraphs
            paragraphs = (List<Map<String, Object>>) article.get("paragraphs");
            for (Map<String, Object> paragraph : paragraphs) {
                contextIndex = sourceTextData.size();
                sourceTextData.add(paragraph.get("context").toString());

                // iterate through the questions
                questions = (List<Map<String, Object>>) paragraph.get("qas");
                for (Map<String, Object> question : questions) {
                    questionIndex = sourceTextData.size();
                    sourceTextData.add(question.get("question").toString());
                    questionInfo = new QuestionInfo(questionIndex, titleIndex, contextIndex);
                    questionInfoList.add(questionInfo);

                    // iterate through the answers
                    answers = (List<Map<String, Object>>) question.get("answers");
                    for (Map<String, Object> answer : answers) {
                        answerIndex = targetTextData.size();
                        targetTextData.add(answer.get("text").toString());
                        questionInfo.addAnswer(answerIndex);
                    }
                }
            }
        }

        preprocess(sourceTextData, true);
        preprocess(targetTextData, false);

        prepared = true;
    }

    /**
     * Gets the {@link Record} for the given index from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param index the index of the requested data item
     * @return a {@link Record} that contains the data and label of the requested data item. The
     *     data {@link NDList} contains three {@link NDArray}s representing the embedded title,
     *     context and question, which are named accordingly. The label {@link NDList} contains
     *     multiple {@link NDArray}s corresponding to each embedded answer.
     */
    @Override
    public Record get(NDManager manager, long index) {
        NDList data = new NDList();
        NDList labels = new NDList();
        QuestionInfo questionInfo = questionInfoList.get(Math.toIntExact(index));

        NDArray title = sourceTextData.getEmbedding(manager, questionInfo.titleIndex);
        title.setName("title");
        NDArray context = sourceTextData.getEmbedding(manager, questionInfo.contextIndex);
        context.setName("context");
        NDArray question = sourceTextData.getEmbedding(manager, questionInfo.questionIndex);
        question.setName("question");

        data.add(title);
        data.add(context);
        data.add(question);

        for (Integer answerIndex : questionInfo.answerIndexList) {
            labels.add(targetTextData.getEmbedding(manager, answerIndex));
        }

        return new Record(data, labels);
    }

    /**
     * Returns the number of records available to be read in this {@code Dataset}. In this
     * implementation, the actual size of available records are the size of {@code
     * questionInfoList}.
     *
     * @return the number of records available to be read in this {@code Dataset}
     */
    @Override
    protected long availableSize() {
        return questionInfoList.size();
    }

    /**
     * Get data from the SQuAD dataset. This method will directly return the whole dataset as an
     * object
     *
     * @return an object of {@link Object} class in the structure of JSON, e.g. {@code Map<String,
     *     List<Map<...>>>}
     */
    @Override
    public Object getData() throws IOException {
        Path usagePath = prepareUsagePath(null);
        Object data;
        try (BufferedReader reader = Files.newBufferedReader(usagePath)) {
            data = JsonUtils.GSON_PRETTY.fromJson(reader, new TypeToken<Object>() {}.getType());
        }
        return data;
    }

    /**
     * Since a question might have no answer, we need extra logic to find the last index of the
     * answer in the {@code TargetTextData}. There are not many consecutive questions without
     * answer, so this logic will not cause a high cost.
     *
     * @param questionInfoIndex the last index of the record in {@code questionInfoList} that needs
     *     to be preprocessed
     * @return the last index of the answer in {@code TargetTextData} that needs to be preprocessed
     */
    private int getLastAnswerIndex(int questionInfoIndex) {
        // Go backwards through the questionInfoList until it finds one with an answer
        for (; questionInfoIndex >= 0; questionInfoIndex--) {
            QuestionInfo questionInfo = questionInfoList.get(questionInfoIndex);
            if (!questionInfo.answerIndexList.isEmpty()) {
                return questionInfo.answerIndexList.get(questionInfo.answerIndexList.size() - 1);
            }
        }

        // Could not find a QuestionInfo with an answer
        return 0;
    }

    /**
     * Performs pre-processing steps on text data such as tokenising, applying {@link
     * ai.djl.modality.nlp.preprocess.TextProcessor}s, creating vocabulary, and word embeddings.
     * Since the record number in this dataset is not equivalent to the length of {@code
     * sourceTextData} and {@code targetTextData}, the limit should be processed.
     *
     * @param newTextData list of all unprocessed sentences in the dataset
     * @param source whether the text data provided is source or target
     * @throws EmbeddingException if there is an error while embedding input
     */
    @Override
    protected void preprocess(List<String> newTextData, boolean source) throws EmbeddingException {
        TextData textData = source ? sourceTextData : targetTextData;
        int index = (int) Math.min(limit, questionInfoList.size()) - 1;
        int lastIndex =
                source ? questionInfoList.get(index).questionIndex : getLastAnswerIndex(index);
        textData.preprocess(manager, newTextData.subList(0, lastIndex + 1));
    }

    /** A builder for a {@link StanfordQuestionAnsweringDataset}. */
    public static class Builder extends TextDataset.Builder<Builder> {

        /** Constructs a new builder. */
        public Builder() {
            artifactId = ARTIFACT_ID;
        }

        /**
         * Returns this {@link Builder} object.
         *
         * @return this {@code BaseBuilder}
         */
        @Override
        public Builder self() {
            return this;
        }

        /**
         * Builds the {@link StanfordQuestionAnsweringDataset}.
         *
         * @return the {@link StanfordQuestionAnsweringDataset}
         */
        public StanfordQuestionAnsweringDataset build() {
            return new StanfordQuestionAnsweringDataset(this);
        }

        MRL getMrl() {
            return repository.dataset(NLP.ANY, groupId, artifactId, VERSION);
        }
    }

    /**
     * This class stores the information of one question. {@code sourceTextData} stores not only the
     * questions, but also the titles and the contexts, and {@code targetTextData} stores right
     * answers and plausible answers. Also, there are some mapping relationships between questions
     * and the other entries, so we need this class to help us assemble the right record.
     */
    private static class QuestionInfo {
        Integer questionIndex;
        Integer titleIndex;
        Integer contextIndex;
        List<Integer> answerIndexList;

        QuestionInfo(Integer questionIndex, Integer titleIndex, Integer contextIndex) {
            this.questionIndex = questionIndex;
            this.titleIndex = titleIndex;
            this.contextIndex = contextIndex;
            this.answerIndexList = new ArrayList<>();
        }

        void addAnswer(Integer answerIndex) {
            this.answerIndexList.add(answerIndex);
        }
    }
}
