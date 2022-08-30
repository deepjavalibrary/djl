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

package ai.djl.examples.inference;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.modality.audio.translator.SpeechRecognitionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;

import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * An example of inference using a speech recognition model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/speech_recognition.md">doc</a>
 * for information about this example.
 */
public final class SpeechRecognition {

    public static final Logger logger = LoggerFactory.getLogger((SpeechRecognition.class));

    private SpeechRecognition() {}

    public static void main(String[] args)
            throws UnsupportedAudioFileException, IOException, TranslateException,
                    ModelNotFoundException, MalformedModelException {
        System.setProperty("ai.djl.default_engine", "PyTorch");
        logger.info("Result: " + predict());
    }

    public static String predict()
            throws UnsupportedAudioFileException, IOException, ModelNotFoundException,
                    MalformedModelException, TranslateException {
        SpeechRecognitionTranslator translator = new SpeechRecognitionTranslator();

        // Load model.
        // Wav2Vec2 model is a speech model that accepts a float array corresponding to the raw
        // waveform of the speech signal.
        Criteria<Audio, String> criteria =
                Criteria.builder()
                        .setTypes(Audio.class, String.class)
                        .optModelUrls(
                                "https://djl-misc.s3.amazonaws.com/tmp/speech_recognition/ai/djl/pytorch/wav2vec2/0.0.1/wav2vec2.ptl.zip")
                        .optTranslator(translator)
                        .optModelName("wav2vec2.ptl")
                        .optEngine("PyTorch")
                        .build();

        // Read in audio file
        Audio audio =
                AudioFactory.getInstance().fromFile(Paths.get("src/test/resources/speech.wav"));
        try (ZooModel<Audio, String> model = criteria.loadModel()) {
            try (Predictor<Audio, String> predictor = model.newPredictor()) {
                return predictor.predict(audio);
            }
        }
    }
}
