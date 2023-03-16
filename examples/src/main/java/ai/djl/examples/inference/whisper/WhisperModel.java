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
package ai.djl.examples.inference.whisper;

import ai.djl.ModelException;
import ai.djl.audio.translator.WhisperTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.bytedeco.ffmpeg.global.avutil;

import java.io.IOException;
import java.nio.file.Path;

/** An example implementation of OpenAI Whisper Model. */
public class WhisperModel implements AutoCloseable {

    ZooModel<Audio, String> whisperModel;

    public WhisperModel() throws ModelException, IOException {
        Criteria<Audio, String> criteria =
                Criteria.builder()
                        .setTypes(Audio.class, String.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/whisper/whisper_en.zip")
                        .optEngine("PyTorch")
                        .optTranslatorFactory(new WhisperTranslatorFactory())
                        .build();
        whisperModel = criteria.loadModel();
    }

    public String speechToText(Audio speech) throws TranslateException {
        try (Predictor<Audio, String> predictor = whisperModel.newPredictor()) {
            return predictor.predict(speech);
        }
    }

    public String speechToText(Path file) throws IOException, TranslateException {
        Audio audio =
                AudioFactory.newInstance()
                        .setChannels(1)
                        .setSampleRate(16000)
                        .setSampleFormat(avutil.AV_SAMPLE_FMT_S16)
                        .fromFile(file);
        return speechToText(audio);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        whisperModel.close();
    }
}
