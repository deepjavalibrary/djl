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

/**
 * A class contains common tasks that can be completed using deep learning.
 *
 * <p>If you view deep learning models as being like a function, then the application is like the
 * function signature. Because there are relatively few signatures used with a lot of research that
 * goes into them, the common signatures are identified by a name. The application is that name.
 */
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
            case "cv/word_recognition":
                return CV.WORD_RECOGNITION;
            case "cv/image_generation":
                return CV.IMAGE_GENERATION;
            case "cv/image_enhancement":
                return CV.IMAGE_ENHANCEMENT;
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

    /** The common set of applications for computer vision (image and video data). */
    public interface CV {

        /** Any computer vision application, including those in {@link CV}. */
        Application ANY = new Application("cv");

        /**
         * An application where images are assigned a single class name.
         *
         * <p>Each image is given one of a fixed number of classes (or a probability of having that
         * one class). The typical signature is Model&lt;{@link ai.djl.modality.cv.Image}, {@link
         * ai.djl.modality.Classifications}&gt;.
         */
        Application IMAGE_CLASSIFICATION = new Application("cv/image_classification");

        /**
         * An application that finds zero or more objects in an image, the object class (see image
         * classification), and their locations as a {@link ai.djl.modality.cv.output.BoundingBox}.
         *
         * <p>The typical signature is Model&lt;{@link ai.djl.modality.cv.Image}, {@link
         * ai.djl.modality.cv.output.DetectedObjects}&gt;.
         *
         * @see <a href="https://d2l.djl.ai/chapter_computer-vision/bounding-box.html">The D2L
         *     chapter on object detection</a>
         */
        Application OBJECT_DETECTION = new Application("cv/object_detection");

        /** An application that classifies each pixel in an image into a category. */
        Application SEMANTIC_SEGMENTATION = new Application("cv/semantic_segmentation");

        /**
         * An application that finds zero or more objects in an image, the object class (see image
         * classification), and their location as a pixel map.
         */
        Application INSTANCE_SEGMENTATION = new Application("cv/instance_segmentation");

        /**
         * An application that accepts an image of a single person and returns the {@link
         * ai.djl.modality.cv.output.Joints} locations of the person.
         *
         * <p>This can often be used with {@link #OBJECT_DETECTION} to identify the people in the
         * image and then run pose estimation on each detected person. The typical signature is
         * Model&lt;{@link ai.djl.modality.cv.Image}, {@link ai.djl.modality.cv.output.Joints}&gt;.
         */
        Application POSE_ESTIMATION = new Application("cv/pose_estimation");

        /**
         * An application that accepts an image or video and classifies the action being done in it.
         */
        Application ACTION_RECOGNITION = new Application("cv/action_recognition");

        /**
         * An application that accepts an image of a single word and returns the {@link String} text
         * of the word.
         *
         * <p>The typical signature is Model&lt;{@link ai.djl.modality.cv.Image}, {@link
         * String}&gt;.
         */
        Application WORD_RECOGNITION = new Application("cv/word_recognition");

        /**
         * An application that accepts a seed and returns generated images.
         *
         * <p>The typical model returns an array of images {@link ai.djl.modality.cv.Image}[].
         */
        Application IMAGE_GENERATION = new Application("cv/image_generation");

        /**
         * An application that accepts an image and returns enhanced images.
         *
         * <p>The typical signature is Model&lt;{@link ai.djl.modality.cv.Image}, {@link
         * ai.djl.modality.cv.Image}&gt;.
         */
        Application IMAGE_ENHANCEMENT = new Application("cv/image_enhancement");
    }

    /** The common set of applications for natural language processing (text data). */
    public interface NLP {

        /** Any NLP application, including those in {@link NLP}. */
        Application ANY = new Application("nlp");

        /**
         * An application that a reference document and a question about the document and returns
         * text answering the question.
         *
         * <p>The typical signature is Model&lt;{@link ai.djl.modality.nlp.qa.QAInput}, {@link
         * String}&gt;.
         */
        Application QUESTION_ANSWER = new Application("nlp/question_answer");

        /**
         * An application that classifies text data.
         *
         * <p>The typical signature is Model&lt;{@link String}, {@link
         * ai.djl.modality.Classifications}&gt;.
         */
        Application TEXT_CLASSIFICATION = new Application("nlp/text_classification");

        /**
         * An application that classifies text into positive or negative, an specific case of {@link
         * #TEXT_CLASSIFICATION}.
         */
        Application SENTIMENT_ANALYSIS = new Application("nlp/sentiment_analysis");

        /**
         * An application that takes a word and returns a feature vector that represents the word.
         *
         * <p>The most representative signature is Model&lt;{@link String}, {@link
         * ai.djl.ndarray.NDArray}&gt;. However, many models will only embed a fixed {@link
         * ai.djl.modality.nlp.Vocabulary} of words. These words are usually given integer indices
         * which may make the signature Model&lt;{@link String}, {@link ai.djl.ndarray.NDArray}&gt;
         * (or {@link ai.djl.ndarray.NDArray}). The signatures may also use singleton {@link
         * ai.djl.ndarray.NDList}s instead of {@link ai.djl.ndarray.NDArray}.
         */
        Application WORD_EMBEDDING = new Application("nlp/word_embedding");

        /**
         * An application that translates text from one language to another.
         *
         * <p>The typical signature is Model&lt;{@link String}, {@link String}&gt;.
         */
        Application MACHINE_TRANSLATION = new Application("nlp/machine_translation");

        /** An application to represent a multiple choice question. */
        Application MULTIPLE_CHOICE = new Application("nlp/multiple_choice");

        /**
         * An application that takes text and returns a feature vector that represents the text.
         *
         * <p>The special case where the text consists of only a word is a {@link #WORD_EMBEDDING}.
         * The typical signature is Model&lt;{@link String}, {@link ai.djl.ndarray.NDArray}&gt;.
         */
        Application TEXT_EMBEDDING = new Application("nlp/text_embedding");
    }

    /** The common set of applications for tabular data. */
    public interface Tabular {

        /** Any tabular application, including those in {@link Tabular}. */
        Application ANY = new Application("tabular");

        /**
         * An application that takes a feature vector (table row) and predicts a numerical feature
         * based on it.
         *
         * @see <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression.html">The D2L
         *     chapter introducing this application</a>
         */
        Application LINEAR_REGRESSION = new Application("tabular/linear_regression");

        /**
         * An application that takes a feature vector (table row) and predicts a categorical feature
         * based on it.
         *
         * <p>There is no typical input, but the typical output is {@link
         * ai.djl.modality.Classifications}.
         *
         * @see <a href="https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html">The D2L
         *     chapter introducing this application</a>
         */
        Application SOFTMAX_REGRESSION = new Application("tabular/softmax_regression");
    }
}
