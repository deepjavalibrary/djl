/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl;

import java.util.Objects;

/** A class contains common deep learning applications. */
public class Application {

    public static final Application UNDEFINED = new Application("undefined");

    private String path;

    Application(String path) {
        this.path = path;
    }

    /**
     * Returns the repository path of the application.
     *
     * @return the repository path of the application
     */
    public String getPath() {
        return path;
    }

    /**
     * Converts a path string to a {@code Application}.
     *
     * @param path the repository path of the application
     * @return the {@code Application}
     */
    public static Application of(String path) {
        switch (path) {
            case "cv":
                return CV.ANY;
            case "cv/image_classification":
                return CV.IMAGE_CLASSIFICATION;
            case "cv/object_detection":
                return CV.OBJECT_DETECTION;
            case "cv/semantic_segmentation":
                return CV.SEMANTIC_SEGMENTATION;
            case "cv/instance_segmentation":
                return CV.INSTANCE_SEGMENTATION;
            case "cv/pose_estimation":
                return CV.POSE_ESTIMATION;
            case "cv/action_recognition":
                return CV.ACTION_RECOGNITION;
            case "nlp":
                return NLP.ANY;
            case "nlp/question_answer":
                return NLP.QUESTION_ANSWER;
            case "nlp/text_classification":
                return NLP.TEXT_CLASSIFICATION;
            case "nlp/sentiment_analysis":
                return NLP.SENTIMENT_ANALYSIS;
            case "nlp/word_embedding":
                return NLP.WORD_EMBEDDING;
            case "nlp/machine_translation":
                return NLP.MACHINE_TRANSLATION;
            case "nlp/multiple_choice":
                return NLP.MULTIPLE_CHOICE;
            case "tabular":
                return Tabular.ANY;
            case "tabular/linear_regression":
                return Tabular.LINEAR_REGRESSION;
            case "undefined":
            default:
                return UNDEFINED;
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return path.replace('/', '.').toUpperCase();
    }

    /**
     * Returns whether this application matches the test application set.
     *
     * @param test a application or application set to test against
     * @return true if it fits within the application set
     */
    public boolean matches(Application test) {
        return path.startsWith(test.path);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Application)) {
            return false;
        }
        return path.equals(((Application) o).path);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(path);
    }

    /** The common set of applications for computer vision. */
    public interface CV {
        Application ANY = new Application("cv");
        Application IMAGE_CLASSIFICATION = new Application("cv/image_classification");
        Application OBJECT_DETECTION = new Application("cv/object_detection");
        Application SEMANTIC_SEGMENTATION = new Application("cv/semantic_segmentation");
        Application INSTANCE_SEGMENTATION = new Application("cv/instance_segmentation");
        Application POSE_ESTIMATION = new Application("cv/pose_estimation");
        Application ACTION_RECOGNITION = new Application("cv/action_recognition");
    }

    /** The common set of applications for natural language processing. */
    public interface NLP {
        Application ANY = new Application("nlp");
        Application QUESTION_ANSWER = new Application("nlp/question_answer");
        Application TEXT_CLASSIFICATION = new Application("nlp/text_classification");
        Application SENTIMENT_ANALYSIS = new Application("nlp/sentiment_analysis");
        Application WORD_EMBEDDING = new Application("nlp/word_embedding");
        Application MACHINE_TRANSLATION = new Application("nlp/machine_translation");
        Application MULTIPLE_CHOICE = new Application("nlp/multiple_choice");
        Application TEXT_EMBEDDING = new Application("nlp/text_embedding");
    }

    /** The common set of applications for tabular data. */
    public interface Tabular {
        Application ANY = new Application("tabular");
        Application LINEAR_REGRESSION = new Application("tabular/linear_regression");
        Application RANDOM_FOREST = new Application("tabular/random_forest");
    }
}
