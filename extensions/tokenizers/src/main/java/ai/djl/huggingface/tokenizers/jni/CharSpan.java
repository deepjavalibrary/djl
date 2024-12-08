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
package ai.djl.huggingface.tokenizers.jni;

/** A class holds character span information. */
public class CharSpan {

    private final int start;
    private final int end;

    /**
     * Constructs a new {@code CharSpan} instance.
     *
     * @param start the start position
     * @param end the end position
     */
    public CharSpan(int start, int end) {
        this.start = start;
        this.end = end;
    }

    /**
     * Returns the start position.
     *
     * @return the start position
     */
    public int getStart() {
        return start;
    }

    /**
     * Returns the end position.
     *
     * @return the end position
     */
    public int getEnd() {
        return end;
    }
}
