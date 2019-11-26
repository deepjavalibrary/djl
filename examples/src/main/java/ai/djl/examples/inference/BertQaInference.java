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

package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.mxnet.zoo.nlp.qa.QAInput;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An example of inference using BertQA.
 *
 * <p>See the <a href="https://github.com/awslabs/djl/blob/master/jupyter/BERTQA.ipynb">jupyter
 * demo</a> with more information about BERT.
 */
public final class BertQaInference {

    private static final Logger logger = LoggerFactory.getLogger(BertQaInference.class);

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        String answer = new BertQaInference().predict();
        logger.info("Answer: {}", answer);
    }

    public String predict() throws IOException, TranslateException, ModelException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel.\n"
                        + "Which operated between December 2004 and April 2006.\n"
                        + "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, paragraph, 384);
        logger.info("Paragraph: {}", input.getParagraph());
        logger.info("Question: {}", input.getQuestion());

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "bert");
        criteria.put("dataset", "book_corpus_wiki_en_uncased");

        try (ZooModel<QAInput, String> model =
                MxModelZoo.BERT_QA.loadModel(criteria, new ProgressBar())) {
            try (Predictor<QAInput, String> predictor = model.newPredictor()) {
                return predictor.predict(input);
            }
        }
    }
}
