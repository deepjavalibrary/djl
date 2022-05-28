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

package ai.djl.audio.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.jtransforms.fft.FloatFFT_1D;

/** Calculate linear spectrogram by short-time fourier transform. */
public class LinearSpecgram implements AudioProcessor {

    static float eps = 1e-14f;

    private float strideMs;
    private float windowsMs;
    private float sampleRate;

    /**
     * Calculate linear spectrogram by short-time fourier transform.
     *
     * @param strideMs Stride size of window
     * @param windowsMs Window size
     * @param sampleRate Sample rate of raw data
     */
    public LinearSpecgram(float strideMs, float windowsMs, int sampleRate) {
        this.strideMs = strideMs;
        this.windowsMs = windowsMs;
        this.sampleRate = sampleRate;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray extractFeatures(NDManager manager, NDArray samples) {
        return stft(samples);
    }

    private NDArray stft(NDArray samples) {
        NDManager manager = samples.getManager();
        int strideSize = (int) (0.001 * sampleRate * strideMs);
        int windowSize = (int) (0.001 * sampleRate * windowsMs);
        long truncateSize = (samples.size() - windowSize) % strideSize;
        long len = samples.size() - truncateSize;
        samples = samples.get(":" + len);

        int rows = ((int) samples.size() - windowSize) / strideSize + 1;

        NDList windowList = new NDList();
        for (int row = 0; row < rows; row++) {
            windowList.add(samples.get(strideSize * row + ":" + (strideSize * row + windowSize)));
        }
        samples = NDArrays.stack(windowList);

        NDArray weighting = hanningWindow(windowSize, manager);
        samples.muli(weighting);
        NDList fftList = new NDList();
        for (int row = 0; row < rows; row++) {
            fftList.add(fft(samples.get(row)));
        }

        NDArray fft = NDArrays.stack(fftList).transpose();
        fft = fft.pow(2);

        weighting = weighting.pow(2);
        NDArray scale = weighting.sum().mul(this.sampleRate);

        NDArray middle = fft.get("1:-1,:");
        middle = middle.mul(2).div(scale);
        NDArray head = fft.get("0,:").div(scale).reshape(1, fft.getShape().get(1));
        NDArray tail = fft.get("-1,:").div(scale).reshape(1, fft.getShape().get(1));
        NDList list = new NDList(head, middle, tail);
        fft = NDArrays.concat(list, 0);

        NDArray freqsArray = manager.arange(fft.getShape().get(0));
        freqsArray = freqsArray.mul(this.sampleRate / windowSize);

        float[] freqs = freqsArray.toFloatArray();
        int ind = 0;
        for (int i = 0; i < freqs.length; i++) {
            if (freqs[i] <= (this.sampleRate / 2)) {
                ind = i;
            } else {
                break;
            }
        }
        ind = ind + 1;

        fft = fft.get(":" + ind + ",:").add(eps);
        fft = fft.log();

        return fft;
    }

    private NDArray fft(NDArray in) {
        float[] rawFFT = in.toFloatArray();
        FloatFFT_1D fft = new FloatFFT_1D(rawFFT.length);
        fft.realForward(rawFFT);
        float[][] result;

        int n = rawFFT.length;
        if (n % 2 == 0) {
            // n is even
            result = new float[2][n / 2 + 1];
            for (int i = 0; i < n / 2; i++) {
                result[0][i] = rawFFT[2 * i]; // the real part fo the fast fourier transform
                result[1][i] =
                        rawFFT[2 * i + 1]; // the imaginary part of the fast fourier transform
            }
            result[1][0] = 0;
            result[0][n / 2] = rawFFT[1];
        } else {
            // n is odd
            result = new float[2][(n + 1) / 2];
            for (int i = 0; i < n / 2; i++) {
                result[0][i] = rawFFT[2 * i]; // the real part fo the fast fourier transform
                result[1][i] =
                        rawFFT[2 * i + 1]; // the imaginary part of the fast fourier transform
            }
            result[1][0] = 0;
            result[1][(n - 1) / 2] = rawFFT[1];
        }

        float[] re = result[0]; // the real part fo the fast fourier transform
        float[] im = result[1]; // the imaginary part of the fast fourier transform
        float[] abs = new float[re.length];
        for (int i = 0; i < re.length; i++) {
            abs[i] = (float) Math.hypot(re[i], im[i]);
        }
        return in.getManager().create(abs);
    }

    private NDArray hanningWindow(int size, NDManager manager) {
        float[] data = new float[size];
        for (int i = 1; i < size; i++) {
            data[i] = (float) (0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1))));
        }
        return manager.create(data);
    }
}
