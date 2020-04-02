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
package ai.djl.modality.nlp.bert;

import ai.djl.modality.nlp.Vocabulary;
import java.io.InputStream;

/** An interface to define BertVocabulary. */
public abstract class BertVocabulary implements Vocabulary {

    /**
     * Parses the vocabulary file and create {@code BertVocabulary}.
     *
     * @param is the input InputStream
     * @return an instance of {@code BertVocabulary}
     */
    static BertVocabulary parse(InputStream is) {
        return null;
    }
}
