package ai.djl.audio.featurizer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class AudioNormalizer implements AudioProcessor {

    private float target_db;
    static float max_gain_db = 300.0f;

    public AudioNormalizer(float target_db) {
        this.target_db = target_db;
    }

    @Override
    public NDArray extractFeatures(NDManager manager, NDArray samples) {
        float gain = target_db - rmsDb(samples);
        gain = Math.min(gain, max_gain_db);

        float factor = (float) Math.pow(10f, gain / 20f);
        samples = samples.mul(factor);
        return samples;
    }

    private float rmsDb(NDArray samples) {
        samples = samples.pow(2).mean().log10().mul(10);
        return samples.toFloatArray()[0];
    }
}
