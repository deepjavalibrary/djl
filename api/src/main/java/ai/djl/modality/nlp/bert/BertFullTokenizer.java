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

import ai.djl.modality.nlp.NlpUtils;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.preprocess.LambdaProcessor;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextCleaner;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.UnicodeNormalizer;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * BertFullTokenizer runs end to end tokenization of input text
 *
 * <p>It will run basic preprocessors to clean the input text and then run {@link
 * WordpieceTokenizer} to split into word pieces.
 *
 * <p>Reference implementation: <a
 * href="https://github.com/google-research/bert/blob/master/tokenization.py#L161">Google Research
 * Bert Tokenizer</a>
 */
public class BertFullTokenizer extends BertTokenizer {

    private Vocabulary vocabulary;
    private List<TextProcessor> basicBertPreprocessors;
    private WordpieceTokenizer wordpieceTokenizer;

    /**
     * Creates an instance of {@code BertFullTokenizer}.
     *
     * @param vocabulary the BERT vocabulary
     * @param lowerCase whether to convert tokens to lowercase
     */
    public BertFullTokenizer(Vocabulary vocabulary, boolean lowerCase) {
        this.vocabulary = vocabulary;
        basicBertPreprocessors = getPreprocessors(lowerCase);
        wordpieceTokenizer = new WordpieceTokenizer(vocabulary, "[UNK]", 200);
    }

    /**
     * Returns the {@link Vocabulary} used for tokenization.
     *
     * @return the {@link Vocabulary} used for tokenization
     */
    public Vocabulary getVocabulary() {
        return vocabulary;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String input) {
        List<String> tokens = new ArrayList<>(Collections.singletonList(input));
        for (TextProcessor processor : basicBertPreprocessors) {
            tokens = processor.preprocess(tokens);
        }
        return wordpieceTokenizer.preprocess(tokens);
    }

    /** {@inheritDoc} */
    @Override
    public String tokenToString(List<String> tokens) {
        return String.join(" ", tokens).replace(" ##", "").trim();
    }

    /**
     * Get a list of {@link TextProcessor}s to process input text for Bert models.
     *
     * @param lowerCase whether to convert input to lowercase
     * @return List of {@code TextProcessor}s
     */
    public static List<TextProcessor> getPreprocessors(boolean lowerCase) {
        List<TextProcessor> processors = new ArrayList<>(10);
        processors.add(new TextCleaner(c -> c == 0 || c == 0xfffd || NlpUtils.isControl(c), '\0'));
        processors.add(new TextCleaner(NlpUtils::isWhiteSpace, ' '));
        processors.add(new LambdaProcessor(String::trim));
        processors.add(new SimpleTokenizer());
        if (lowerCase) {
            processors.add(new LowerCaseConvertor());
        }
        processors.add(new UnicodeNormalizer(Normalizer.Form.NFD));
        processors.add(
                new TextCleaner(c -> Character.getType(c) == Character.NON_SPACING_MARK, '\0'));
        processors.add(new PunctuationSeparator());
        processors.add(new LambdaProcessor(String::trim));
        return processors;
    }
}
