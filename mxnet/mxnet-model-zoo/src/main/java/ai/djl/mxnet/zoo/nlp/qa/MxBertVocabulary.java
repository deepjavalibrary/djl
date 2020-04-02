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
package ai.djl.mxnet.zoo.nlp.qa;

import ai.djl.modality.nlp.bert.BertVocabulary;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/** A MXNet implementaion of BertVocabulary. */
public class MxBertVocabulary extends BertVocabulary {

    private static final Gson GSON = new GsonBuilder().create();

    @SerializedName("token_to_idx")
    private Map<String, Long> token2idx;

    @SerializedName("idx_to_token")
    private List<String> idx2token;

    /**
     * Parses the vocabulary file and create {@code MxBertVocabulary}.
     *
     * @param is the input InputStream of the vocabulary file
     * @return an instance of {@code MxBertVocabulary}
     */
    public static MxBertVocabulary parse(InputStream is) {
        try (Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            return GSON.fromJson(reader, MxBertVocabulary.class);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        return (token2idx.containsKey(token)) ? token2idx.get(token) : token2idx.get("[UNK]");
    }

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        return idx2token.get((int) index);
    }
}
