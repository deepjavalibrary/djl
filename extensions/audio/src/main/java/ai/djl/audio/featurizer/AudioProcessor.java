package ai.djl.audio.featurizer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public interface AudioProcessor {
    NDArray extractFeatures(NDManager manager, NDArray samples);
}
