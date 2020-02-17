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
public interface Application {

    /** The common set of applications for computer vision. */
    enum CV {
        IMAGE_CLASSIFICATION,
        OBJECT_DETECTION,
        SEMANTIC_SEGMENTATION,
        INSTANCE_SEGMENTATION,
        POSE_ESTIMATION,
        ACTION_RECOGNITION
    }

    /** The common set of applications for natural language processing. */
    enum NLP {
        QUESTION_ANSWER,
        TEXT_CLASSIFICATION
    }
}
