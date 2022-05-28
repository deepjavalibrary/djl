package ai.djl.audio;

import ai.djl.ndarray.NDArray;

public class AudioUtils {

    public static float rmsDb(NDArray samples) {
        samples = samples.pow(2).mean().log10().mul(10);
        return samples.toFloatArray()[0];
    }
}
