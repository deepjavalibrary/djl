package ai.djl.modality.audio;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public interface AudioFeaturizer {
    default NDArray ExtractFeatures(NDManager manager, NDArray array) throws Exception{
        return null;
    }
}
