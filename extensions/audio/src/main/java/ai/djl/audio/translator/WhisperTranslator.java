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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A {@link Translator} that process the {@link Audio} into {@link String} to get a text translation
 * of the audio.
 */
public class WhisperTranslator implements NoBatchifyTranslator<Audio, String> {

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
        List<String> sentence = new ArrayList<>();
        for (long ele : result.toLongArray()) {
            sentence.add(vocabulary.getToken(ele));
            if ("<|endoftext|>".equals(vocabulary.getToken(ele))) {
                break;
            }
        }
        String output = String.join(" ", sentence);
        return output.replaceAll("[^a-zA-Z0-9<|> ,.!]", "");
    }
}
