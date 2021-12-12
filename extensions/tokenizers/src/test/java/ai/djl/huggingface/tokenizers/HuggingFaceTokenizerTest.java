/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.huggingface.tokenizers;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

public class HuggingFaceTokenizerTest {

    @Test
    public void testTokenizer() {
        String input = "Hello, y'all! How are you 😁 ?";
        String[] inputs = {"Hello, y'all!", "How are you 😁 ?"};

        String[] expected = {
            "[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"
        };

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("bert-base-cased")) {
            List<String> ret = tokenizer.tokenize(input);
            Assert.assertEquals(ret.toArray(new String[0]), expected);
            Encoding encoding = tokenizer.encode(input);

            long[] ids = {101, 8667, 117, 194, 112, 1155, 106, 1731, 1132, 1128, 100, 136, 102};
            long[] typeIds = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            long[] wordIds = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1};
            long[] attentionMask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            long[] specialTokenMask = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

            Assert.assertEquals(expected, encoding.getTokens());
            Assert.assertEquals(ids, encoding.getIds());
            Assert.assertEquals(typeIds, encoding.getTypeIds());
            Assert.assertEquals(wordIds, encoding.getWordIds());
            Assert.assertEquals(attentionMask, encoding.getAttentionMask());
            Assert.assertEquals(specialTokenMask, encoding.getSpecialTokenMask());

            Encoding[] encodings = tokenizer.batchEncode(Arrays.asList(inputs));
            Assert.assertEquals(encodings.length, 2);
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("addSpecialTokens", "false");
        try (HuggingFaceTokenizer tokenizer =
                HuggingFaceTokenizer.newInstance("bert-base-cased", options)) {
            String[] exp = {"Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"};

            Encoding encoding = tokenizer.encode(Arrays.asList(inputs));
            Assert.assertEquals(encoding.getTokens(), exp);
        }
    }
}
