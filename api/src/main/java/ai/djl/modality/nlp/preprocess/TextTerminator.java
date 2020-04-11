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
package ai.djl.modality.nlp.preprocess;

import java.util.ArrayList;
import java.util.List;

/** A {@link TextProcessor} that adds a beginning of string and end of string token. */
public class TextTerminator implements TextProcessor {

    private static final String DEFAULT_EOS_TOKEN = "<eos>";
    private static final String DEFAULT_BOS_TOKEN = "<bos>";

    private boolean addBosToken;
    private boolean addEosToken;
    private String eosToken;
    private String bosToken;

    /** Constructs a default {@link TextTerminator}. */
    public TextTerminator() {
        this(true, true);
    }

    /**
     * Constructs a {@link TextTerminator} using the default tokens.
     *
     * @param addBosToken true to add a beginning of text token
     * @param addEosToken true to add an end of text token
     */
    public TextTerminator(boolean addBosToken, boolean addEosToken) {
        this.addBosToken = addBosToken;
        this.addEosToken = addEosToken;
        this.bosToken = DEFAULT_BOS_TOKEN;
        this.eosToken = DEFAULT_EOS_TOKEN;
    }

    /**
     * Constructs a {@link TextTerminator}.
     *
     * @param addBosToken true to add a beginning of text token
     * @param addEosToken true to add an end of text token
     * @param bosToken the token to add to the beginning of the text
     * @param eosToken the token to add to the end of the text
     */
    public TextTerminator(
            boolean addBosToken, boolean addEosToken, String bosToken, String eosToken) {
        this.addBosToken = addBosToken;
        this.addEosToken = addEosToken;
        this.bosToken = bosToken;
        this.eosToken = eosToken;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        List<String> list = new ArrayList<>(tokens.size() + 2);
        if (addBosToken) {
            list.add(bosToken);
        }
        list.addAll(tokens);
        if (addEosToken) {
            list.add(eosToken);
        }
        return list;
    }
}
