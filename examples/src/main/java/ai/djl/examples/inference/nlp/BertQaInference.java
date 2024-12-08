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

package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * An example of inference using BertQA.
 *
 * <p>See:
 *
 * <ul>
 *   <li>the <a href="https://docs.djl.ai/master/docs/demos/jupyter/BERTQA.html">jupyter demo</a>
 *       with more information about BERT.
 *   <li>the <a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/BERT_question_and_answer.md">docs</a>
 *       for information about running this example.
 * </ul>
 */
public final class BertQaInference {

    private static final Logger logger = LoggerFactory.getLogger(BertQaInference.class);

    private BertQaInference() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        String answer = BertQaInference.predict();
        logger.info("Output: {}", answer);
    }

    public static String predict() throws IOException, TranslateException, ModelException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, paragraph);
        logger.info("Paragraph: {}", input.getParagraph());
        logger.info("Question: {}", input.getQuestion());

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .optArgument("detail", true)
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            return predictor.predict(input);
        }
    }
}
