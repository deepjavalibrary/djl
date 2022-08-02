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
import ai.djl.modality.audio.translator.SpeechRecognitionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * An example of inference using a speech recognition model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/speech_recognition.md">doc</a>
 * for information about this example.
 */
public final class SpeechRecognition {

    public static final Logger logger =
            LoggerFactory.getLogger((SpeechRecognitionTranslator.class));

    private SpeechRecognition() {}

    public static void main(String[] args)
            throws UnsupportedAudioFileException, IOException, TranslateException,
                    ModelNotFoundException, MalformedModelException {
        SpeechRecognition.predict();
    }

    public static String predict()
            throws UnsupportedAudioFileException, IOException, ModelNotFoundException,
                    MalformedModelException, TranslateException {

        String url =
                "https://djl-misc.s3.amazonaws.com/tmp/speech_recognition/ai/djl/pytorch/wav2vec2/0.0.1/wav2vec2.ptl.zip";
        Map<String, String> arguments = new ConcurrentHashMap<>();
        SpeechRecognitionTranslator translator =
                SpeechRecognitionTranslator.builder(arguments).build();

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optTranslator(translator)
                        .optModelName("wav2vec2.ptl")
                        .optEngine("PyTorch")
                        .build();

        File f = new File("src/test/resources/SpeechRecognition_scent_of_a_woman_future.wav");

        // read in audio file
        NDList list;
        try (NDManager manager = NDManager.newBaseManager()) {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(f);
            int bytesPerFrame = audioInputStream.getFormat().getFrameSize();
            // Set an arbitrary buffer size of 1024 frames.
            int numBytes = 1024 * bytesPerFrame;
            byte[] audioBytes = new byte[numBytes];
            int numBytesRead;
            int numFramesRead;
            int totalFramesRead = 0;
            // Try to read numBytes bytes from the file.
            while ((numBytesRead = audioInputStream.read(audioBytes)) != -1) {
                // Calculate the number of frames actually read.
                numFramesRead = numBytesRead / bytesPerFrame;
                totalFramesRead += numFramesRead;
            }
            NDArray array = manager.create(audioBytes);
            list = new NDList(array);
        }

        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            try (Predictor<NDList, NDList> predictor = model.newPredictor()) {
                return predictor.predict(list).get(0).toString();
            }
        }
    }
}
