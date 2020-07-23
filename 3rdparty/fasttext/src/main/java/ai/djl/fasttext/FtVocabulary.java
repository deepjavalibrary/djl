/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext;

import ai.djl.modality.nlp.Vocabulary;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link FtVocabulary} is an implementation of {@link Vocabulary} used in conjunction with {@link
 * FtWord2VecWordEmbedding}.
 */
public class FtVocabulary implements Vocabulary {

    private Map<Long, String> tokenMap = new ConcurrentHashMap<>();
    private Map<String, Long> indicesMap = new ConcurrentHashMap<>();
    private long index;

    @Override
    public String getToken(long index) {
        return tokenMap.get(index);
    }

    @Override
    public long getIndex(String token) {
        if (!indicesMap.containsKey(token)) {
            indicesMap.put(token, index);
            tokenMap.put(index++, token);
        }
        return indicesMap.get(token);
    }

    @Override
    public long size() {
        return indicesMap.size();
    }
}
