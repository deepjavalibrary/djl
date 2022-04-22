/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.basicdataset.nlp;

/**
 * An enum class for Universal POS tags which mark the core part-of-speech categories.
 *
 * @see <a href="https://universaldependencies.org/u/pos/">Universal POS tags</a>
 */
public enum UniversalPosTag {
    ADJ,
    ADV,
    INTJ,
    NOUN,
    PROPN,
    VERB,
    ADP,
    AUX,
    CCONJ,
    DET,
    NUM,
    PART,
    PRON,
    SCONJ,
    PUNCT,
    SYM,
    X;
}
