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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.audio.Audio;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** An example implementation of OpenAI Whisper Model. */
public class WhisperModel implements AutoCloseable {

    ZooModel<NDList, NDList> whisperModel;
    Predictor<Audio, String> whisper;

    public WhisperModel() throws ModelException, IOException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(
                                "https://resources.djl.ai/demo/pytorch/whisper/whisper_en.zip")
                        .optEngine("PyTorch")
                        .build();
        whisperModel = criteria.loadModel();
        whisper = whisperModel.newPredictor(new WhisperTranslator(whisperModel.getNDManager()));
    }

    public String speechToText(Audio speech) throws TranslateException {
        return whisper.predict(speech);
    }

    public String speechToText(Path file) throws IOException, TranslateException {
        float[] result =
                getStreamFloats(file.toAbsolutePath().toString(), "s16le", "pcm_s16le", 16000);
        return speechToText(new Audio(result));
    }

    private float[] getStreamFloats(String filePath, String format, String codec, int samplingRate)
            throws IOException {
        String cmd =
                String.join(
                        " ",
                        "ffmpeg -nostdin -threads 0",
                        "-i",
                        filePath,
                        "-f",
                        format,
                        "-acodec",
                        codec,
                        "-ac 1",
                        "-ar",
                        String.valueOf(samplingRate),
                        "-");
        Process process = new ProcessBuilder("sh", "-c", cmd).start();
        Thread thread =
                new Thread(
                        () -> {
                            try {
                                Utils.toString(process.getErrorStream());
                            } catch (IOException e) {
                                throw new RuntimeException(e); // NOPMD
                            }
                        });
        thread.start();
        byte[] array = Utils.toByteArray(process.getInputStream());
        float[] result = new float[array.length / 2];
        List<Short> data = new ArrayList<>();
        ByteBuffer bb = ByteBuffer.wrap(array);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        while (bb.hasRemaining()) {
            data.add(bb.getShort());
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = data.get(i) / 32768.0f;
        }
        return result;
    }

    @Override
    public void close() {
        whisper.close();
        whisperModel.close();
    }
}
