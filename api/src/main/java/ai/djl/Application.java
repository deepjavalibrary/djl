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

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return path.replace('/', '.').toUpperCase();
    }

    /** The common set of applications for computer vision. */
    public interface CV {
        Application IMAGE_CLASSIFICATION = new Application("cv/image_classification");
        Application OBJECT_DETECTION = new Application("cv/object_detection");
        Application SEMANTIC_SEGMENTATION = new Application("cv/semantic_segmentation");
        Application INSTANCE_SEGMENTATION = new Application("cv/instance_segmentation");
        Application POSE_ESTIMATION = new Application("cv/pose_estimation");
        Application ACTION_RECOGNITION = new Application("cv/action_recognition");
    }

    /** The common set of applications for natural language processing. */
    public interface NLP {
        Application QUESTION_ANSWER = new Application("nlp/question_answer");
        Application TEXT_CLASSIFICATION = new Application("nlp/text_classification");
        Application SENTIMENT_ANALYSIS = new Application("nlp/sentiment_analysis");
        Application WORD_EMBEDDING = new Application("nlp/word_embedding");
        Application MACHINE_TRANSLATION = new Application("nlp/machine_translation");
        Application MULTIPLE_CHOICE = new Application("nlp/multiple_choice");
    }

    /** The common set of applications for tabular data. */
    public interface Tabular {
        Application LINEAR_REGRESSION = new Application("tabular/linear_regression");
    }
}
