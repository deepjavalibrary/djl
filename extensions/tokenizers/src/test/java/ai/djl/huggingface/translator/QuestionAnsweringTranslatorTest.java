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
package ai.djl.huggingface.translator;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

/** Tests for {@link QuestionAnsweringTranslator}. */
public class QuestionAnsweringTranslatorTest {

    @Test
    public void testNormalInput() throws ModelException, IOException, TranslateException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, paragraph);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            String answer = predictor.predict(input);
            Assert.assertNotNull(answer);
            Assert.assertTrue(
                    answer.toLowerCase().contains("december")
                            || answer.toLowerCase().contains("2004"));
        }
    }

    @Test
    public void testNormalInputWithDetail() throws ModelException, IOException, TranslateException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, paragraph);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .optArgument("detail", true)
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            String answer = predictor.predict(input);
            Assert.assertNotNull(answer);
            // Should return JSON with score, start, end, answer
            Assert.assertTrue(answer.contains("score"));
            Assert.assertTrue(answer.contains("answer"));
        }
    }

    @Test
    public void testMinimalParagraphWithDetail() throws ModelException, IOException {
        // Use a very short paragraph that's unlikely to contain the answer
        String question = "What is the exact date and time of the event?";
        String paragraph = "Yes.";

        QAInput input = new QAInput(question, paragraph);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .optArgument("detail", true)
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            // Should not crash with ArithmeticException (division by zero)
            // May return an answer or throw exception, but should handle gracefully
            String answer = predictor.predict(input);
            // If it returns an answer, verify it's from the paragraph, not the question
            Assert.assertNotNull(answer);
            Assert.assertFalse(
                    answer.toLowerCase().contains("date")
                            || answer.toLowerCase().contains("time")
                            || answer.toLowerCase().contains("event"),
                    "Answer should not contain question text");
        } catch (TranslateException e) {
            // If it throws an exception, verify it's not ArithmeticException
            Assert.assertFalse(
                    e.getCause() instanceof ArithmeticException,
                    "Should not throw ArithmeticException (division by zero)");
        }
    }

    @Test
    public void testMinimalParagraphWithoutDetail() throws ModelException, IOException {
        // Use a very short paragraph
        String question = "What is the exact date and time of the event?";
        String paragraph = "Yes.";

        QAInput input = new QAInput(question, paragraph);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            // Should handle gracefully without crashing
            String answer = predictor.predict(input);
            // If it returns an answer, verify it's from the paragraph, not the question
            Assert.assertNotNull(answer);
            Assert.assertFalse(
                    answer.toLowerCase().contains("date")
                            || answer.toLowerCase().contains("time")
                            || answer.toLowerCase().contains("event"),
                    "Answer should not contain question text");
        } catch (TranslateException e) {
            // Acceptable to throw exception for invalid input
            Assert.assertTrue(true, "Gracefully handled invalid input");
        }
    }

    @Test
    public void testQuestionNotReturnedAsAnswer()
            throws ModelException, IOException, TranslateException {
        // Question contains information that should NOT be returned as answer
        String question = "What is the secret key abc123xyz?";
        String paragraph = "The system uses various authentication keys for security.";

        QAInput input = new QAInput(question, paragraph);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelUrls(
                                "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            String answer = predictor.predict(input);
            Assert.assertNotNull(answer);
            // Answer should NOT contain the secret key from the question
            Assert.assertFalse(
                    answer.contains("abc123xyz"), "Answer should not contain text from question");
            // Answer should come from the paragraph
            Assert.assertTrue(
                    answer.toLowerCase().contains("key")
                            || answer.toLowerCase().contains("authentication")
                            || answer.toLowerCase().contains("security"));
        }
    }
}
