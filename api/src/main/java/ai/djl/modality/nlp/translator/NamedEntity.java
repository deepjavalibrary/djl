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
package ai.djl.modality.nlp.translator;

import ai.djl.util.JsonUtils;

/** A class that represents a {@code NamedEntity} object. */
public class NamedEntity {

    private String entity;
    private float score;
    private int index;
    private String word;
    private int start;
    private int end;

    /**
     * Constructs a new instance of {@code NamedEntity}.
     *
     * @param entity the class of the entity
     * @param score the score of the entity
     * @param index the position of the entity in the original sentence
     * @param word the token of the entity
     * @param start the start index of the word in the sentence
     * @param end the end index of the word in the sentence
     */
    public NamedEntity(String entity, float score, int index, String word, int start, int end) {
        this.entity = entity;
        this.score = score;
        this.index = index;
        this.word = word;
        this.start = start;
        this.end = end;
    }

    /**
     * Returns the class of the entity.
     *
     * @return the class of the entity
     */
    public String getEntity() {
        return entity;
    }

    /**
     * Returns the score of the entity.
     *
     * @return the score of the entity
     */
    public float getScore() {
        return score;
    }

    /**
     * Returns the position of the entity in the original sentence.
     *
     * @return the position of the entity in the original sentence
     */
    public int getIndex() {
        return index;
    }

    /**
     * Returns the token of the entity.
     *
     * @return the token of the entity
     */
    public String getWord() {
        return word;
    }

    /**
     * Returns the start index of the word in the sentence.
     *
     * @return the start index of the word in the sentence
     */
    public int getStart() {
        return start;
    }

    /**
     * Returns the end index of the word in the sentence.
     *
     * @return the end index of the word in the sentence
     */
    public int getEnd() {
        return end;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return JsonUtils.GSON_PRETTY.toJson(this);
    }
}
