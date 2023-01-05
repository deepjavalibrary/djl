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
package ai.djl.audio.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/** Apply Log Mel spectrogram to the given data. */
public class LogMelSpectrogram implements AudioProcessor {

    private static final int N_FFT = 400;
    private static final int HOP_LENGTH = 160;

    private NDArray melFilters;

    /**
     * Constructs a new instance of {@code LogMelSpectrogram}.
     *
     * @param melFilter the mel filter
     */
    public LogMelSpectrogram(NDArray melFilter) {
        this.melFilters = melFilter;
    }

    /**
     * Loads the mel filterbank matrix for projecting STFT into a Mel spectrogram.
     *
     * <p>Allows decoupling librosa dependency; saved using: np.savez_compressed( "mel_filters.npz",
     * mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80), )
     *
     * @param melFile the mdel file saved in .npz format
     * @param numMel number of mel
     * @param manager manager to set for the content
     * @return a new instance of {@code LogMelSpectrogram}
     * @throws IOException file not loadable
     */
    public static LogMelSpectrogram newInstance(Path melFile, int numMel, NDManager manager)
            throws IOException {
        try (InputStream is = Files.newInputStream(melFile)) {
            return newInstance(is, numMel, manager);
        }
    }

    /**
     * Loads the mel filterbank matrix for projecting STFT into a Mel spectrogram.
     *
     * @param is the input stream
     * @param numMel number of mel
     * @param manager manager to set for the content
     * @return a new instance of {@code LogMelSpectrogram}
     * @throws IOException file not loadable
     */
    public static LogMelSpectrogram newInstance(InputStream is, int numMel, NDManager manager)
            throws IOException {
        return new LogMelSpectrogram(NDList.decode(manager, is).get("mel_" + numMel));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray extractFeatures(NDManager manager, NDArray samples) {
        NDArray window = manager.hanningWindow(N_FFT);
        NDArray stft = samples.stft(N_FFT, HOP_LENGTH, true, window, true);
        NDArray magnitudes = stft.get(":,:-1").abs().pow(2);
        NDArray melSpec = melFilters.matMul(magnitudes);
        melSpec.attach(manager);
        NDArray logSpec = melSpec.clip(1e-10, Float.MAX_VALUE).log10();
        logSpec = logSpec.maximum(logSpec.max().sub(8.0f));
        logSpec = logSpec.add(4.0f).div(4.0f);
        return logSpec;
    }
}
