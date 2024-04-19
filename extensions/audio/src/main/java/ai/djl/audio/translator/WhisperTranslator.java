/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.audio.translator;

import ai.djl.audio.processor.AudioProcessor;
import ai.djl.audio.processor.LogMelSpectrogram;
import ai.djl.audio.processor.PadOrTrim;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link Translator} that process the {@link Audio} into {@link String} to get a text translation
 * of the audio.
 */
public class WhisperTranslator implements NoBatchifyTranslator<Audio, String> {

    private static final Map<Character, Byte> BYTES_DECODER = bpeDecoder();
    private List<AudioProcessor> processors;
    private Vocabulary vocabulary;

    /** Constructs a new instance of {@code WhisperTranslator}. */
    public WhisperTranslator() {
        processors = new ArrayList<>();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        Path melFile = path.resolve("mel_80_filters.npz");

        processors.add(new PadOrTrim(480000));
        // Use model's NDManager
        NDManager modelManager = ctx.getModel().getNDManager();
        processors.add(LogMelSpectrogram.newInstance(melFile, 80, modelManager));

        Map<String, Integer> vocab;
        Map<String, Integer> added;
        Type type = new TypeToken<Map<String, Integer>>() {}.getType();
        try (Reader reader = Files.newBufferedReader(path.resolve("vocab.json"))) {
            vocab = JsonUtils.GSON.fromJson(reader, type);
        }
        try (Reader reader = Files.newBufferedReader(path.resolve("added_tokens.json"))) {
            added = JsonUtils.GSON.fromJson(reader, type);
        }
        String[] result = new String[vocab.size() + added.size()];
        vocab.forEach((key, value) -> result[value] = key);
        added.forEach((key, value) -> result[value] = key);
        vocabulary = new DefaultVocabulary(Arrays.asList(result));
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Audio input) throws Exception {
        NDArray samples = ctx.getNDManager().create(input.getData());
        for (AudioProcessor processor : processors) {
            samples = processor.extractFeatures(samples.getManager(), samples);
        }
        samples = samples.expandDims(0);
        NDArray placeholder = ctx.getNDManager().create("");
        placeholder.setName("module_method:generate");
        return new NDList(samples, placeholder);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray result = list.singletonOrThrow();
        StringBuilder sb = new StringBuilder();
        for (long ele : result.toLongArray()) {
            sb.append(vocabulary.getToken(ele));
            if ("<|endoftext|>".equals(vocabulary.getToken(ele))) {
                break;
            }
        }
        byte[] buf = new byte[sb.length()];
        for (int i = 0; i < sb.length(); ++i) {
            char c = sb.charAt(i);
            buf[i] = BYTES_DECODER.get(c);
        }

        return new String(buf, StandardCharsets.UTF_8);
    }

    /**
     * Returns list of utf-8 byte and a mapping to unicode strings.
     *
     * <p>We specifically avoids mapping to whitespace/control characters the bpe code barfs on. The
     * reversible bpe codes work on unicode strings. This means you need a large # of unicode
     * characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
     * dataset you end up needing around 5K for decent coverage. This is a significant percentage of
     * your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
     * unicode strings.
     */
    private static Map<Character, Byte> bpeDecoder() {
        Map<Character, Byte> map = new ConcurrentHashMap<>();
        for (char i = '!'; i <= '~'; ++i) {
            map.put(i, (byte) i);
        }
        for (char i = '¡'; i <= '¬'; ++i) {
            map.put(i, (byte) i);
        }
        for (char i = '®'; i <= 'ÿ'; ++i) {
            map.put(i, (byte) i);
        }

        int n = 0;
        for (char i = 0; i < 256; ++i) {
            if (!map.containsKey(i)) {
                map.put((char) (256 + n), (byte) i);
                ++n;
            }
        }
        return map;
    }
}
