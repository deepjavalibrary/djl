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
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * WordpieceTokenizer tokenizes a piece of text into its word pieces.
 *
 * <p>This uses a greedy longest-match-first algorithm to perform tokenization using the given
 * vocabulary. The input text should already be cleaned and preprocessed.
 *
 * <pre>
 * jshell&gt; String input = "unaffable";
 * jshell&gt; wordpieceTokenizer.tokenize(intput);
 * ["un", "##aff", "##able"]
 * </pre>
 *
 * <p>Reference implementation: <a
 * href="https://github.com/google-research/bert/blob/master/tokenization.py#L300">Google Research
 * Bert Tokenizer</a>
 */
public class WordpieceTokenizer extends SimpleTokenizer {

    private String unknown;
    private int maxInputChars;
    private Vocabulary vocabulary;

    /**
     * Creates an instance of {@code WordpieceTokenizer}.
     *
     * @param vocabulary a {@code DefaultVocabulary} used for wordpiece tokenization
     * @param unknown String that represent unknown token
     * @param maxInputChars maximum number of input characters
     */
    public WordpieceTokenizer(Vocabulary vocabulary, String unknown, int maxInputChars) {
        this.unknown = unknown;
        this.maxInputChars = maxInputChars;
        this.vocabulary = vocabulary;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String sentence) {
        StringBuilder sb = new StringBuilder();
        List<String> subTokens = new ArrayList<>();
        List<String> outputTokens = new ArrayList<>();
        for (String token : super.tokenize(sentence.trim())) {
            char[] chars = token.toCharArray();
            if (chars.length > maxInputChars) {
                outputTokens.add(unknown);
                continue;
            }
            boolean isBad = false;
            int start = 0;
            subTokens.clear();
            String currentSubString = null;
            while (start < chars.length) {
                int end = chars.length;
                while (start < end) {
                    sb.setLength(0);
                    sb.append(token, start, end);
                    if (start > 0) {
                        sb.insert(0, "##");
                    }
                    String subString = sb.toString();
                    if (vocabulary.contains(subString)) {
                        currentSubString = subString;
                        break;
                    } else {
                        currentSubString = null;
                    }
                    end--;
                }
                if (currentSubString == null) {
                    isBad = true;
                    break;
                }
                subTokens.add(currentSubString);
                if (subTokens.size() > maxInputChars) {
                    throw new IllegalStateException("Too many subTokens for: '" + sentence + '\'');
                }
                start = end;
            }
            if (isBad) {
                outputTokens.add(unknown);
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }
}
