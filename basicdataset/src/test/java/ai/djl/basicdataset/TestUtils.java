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
package ai.djl.basicdataset;

import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;

public final class TestUtils {

    private TestUtils() {}

    public static TextEmbedding getTextEmbedding(NDManager manager, int embeddingSize) {
        return new TextEmbedding() {

            /** {@inheritDoc} */
            @Override
            public long[] preprocessTextToEmbed(List<String> text) {
                return new long[text.size()];
            }

            /** {@inheritDoc} */
            @Override
            public NDArray embedText(NDManager manager, long[] textIndices) {
                return manager.zeros(new Shape(textIndices.length, embeddingSize));
            }

            /** {@inheritDoc} */
            @Override
            public NDArray embedText(NDArray textIndices) {
                return null;
            }

            /** {@inheritDoc} */
            @Override
            public List<String> unembedText(NDArray textEmbedding) {
                return null;
            }
        };
    }
}
