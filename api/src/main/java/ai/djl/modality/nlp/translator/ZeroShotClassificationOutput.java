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
package ai.djl.modality.nlp.translator;

/** A class that represents a {@code ZeroShotClassificationOutput} object. */
public class ZeroShotClassificationOutput {

    private String sequence;
    private String[] labels;
    private double[] scores;

    /**
     * Constructs a new {@code ZeroShotClassificationOutput} instance.
     *
     * @param sequence the input text
     * @param labels the labels
     * @param scores the scores of the labels
     */
    public ZeroShotClassificationOutput(String sequence, String[] labels, double[] scores) {
        this.sequence = sequence;
        this.labels = labels;
        this.scores = scores;
    }

    /**
     * Returns the input text.
     *
     * @return the input text
     */
    public String getSequence() {
        return sequence;
    }

    /**
     * Returns the labels in sorted order.
     *
     * @return the labels in sorted order
     */
    public String[] getLabels() {
        return labels;
    }

    /**
     * Returns the scores of the labels.
     *
     * @return the scores of the labels
     */
    public double[] getScores() {
        return scores;
    }
}
