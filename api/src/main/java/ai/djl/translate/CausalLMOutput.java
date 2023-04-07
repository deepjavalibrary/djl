package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.util.NativeResource;

public class CausalLMOutput {

    // [batch, seq, feature]
    public NDArray logits;

    // (k, v) * numLayer, k or v: [batch, heads, seq, feature]
    public NativeResource<Long> pastKeyValues;

    public CausalLMOutput(NDArray logits, NativeResource<Long> paskKeyValues) {
        this.logits = logits;
        this.pastKeyValues = paskKeyValues;
    }
}
