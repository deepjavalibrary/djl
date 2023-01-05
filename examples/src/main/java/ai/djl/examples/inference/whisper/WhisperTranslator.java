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
package ai.djl.examples.inference.whisper;

import ai.djl.audio.processor.AudioProcessor;
import ai.djl.audio.processor.LogMelSpectrogram;
import ai.djl.audio.processor.PadOrTrim;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class WhisperTranslator implements NoBatchifyTranslator<Audio, String> {

    List<AudioProcessor> processors;
    Vocabulary vocabulary;

    public WhisperTranslator(NDManager manager) throws IOException {
        DownloadUtils.download(
                "https://resources.djl.ai/audios/mel_80_filters.npz",
                "build/test/mel_80_filters.npz");
        DownloadUtils.download(
                "https://resources.djl.ai/demo/pytorch/whisper/vocab.json",
                "build/test/vocab.json");
        DownloadUtils.download(
                "https://resources.djl.ai/demo/pytorch/whisper/added_tokens.json",
                "build/test/added_tokens.json");

        processors = new ArrayList<>();
        processors.add(new PadOrTrim(480000));
        processors.add(
                new LogMelSpectrogram(Paths.get("build/test/mel_80_filters.npz"), 80, manager));

        Map<String, Integer> vocab = // NOPMD
                new Gson()
                        .fromJson(
                                Files.newBufferedReader(Paths.get("build/test/vocab.json")),
                                new TypeToken<Map<String, Integer>>() {}.getType());
        Map<String, Integer> added = // NOPMD
                new Gson()
                        .fromJson(
                                Files.newBufferedReader(Paths.get("build/test/added_tokens.json")),
                                new TypeToken<Map<String, Integer>>() {}.getType());
        String[] result = new String[vocab.size() + added.size()];
        vocab.forEach((key, value) -> result[value] = key);
        added.forEach((key, value) -> result[value] = key);
        vocabulary = new DefaultVocabulary(Arrays.asList(result));
    }

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
