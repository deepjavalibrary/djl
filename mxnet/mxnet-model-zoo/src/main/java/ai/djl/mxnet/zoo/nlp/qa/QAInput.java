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
package ai.djl.mxnet.zoo.nlp.qa;

/** The input container for BERT QA model. */
public class QAInput {

    private String question;
    private String paragraph;
    private int seqLength;

    /**
     * Creates the BERT QA model.
     *
     * @param question the question for the model
     * @param paragraph the resource document that contains the answer
     * @param seqLength the maximum length of the sentence
     */
    public QAInput(String question, String paragraph, int seqLength) {
        this.question = question;
        this.paragraph = paragraph;
        this.seqLength = seqLength;
    }

    /**
     * Gets the question for the model.
     *
     * @return the question for the model
     */
    public String getQuestion() {
        return question;
    }

    /**
     * Gets the resource document that contains the answer.
     *
     * @return the resource document that contains the answer
     */
    public String getParagraph() {
        return paragraph;
    }

    /**
     * Gets the sequence length.
     *
     * @return the sequence length
     */
    public int getSeqLength() {
        return seqLength;
    }
}
