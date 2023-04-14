package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.NativeResource;

/** CausalLMOuput is used to contain multiple output of a language model. */
public class CausalLMOutput {

    // [batch, seq, feature]
    public NDArray logits;

    // (k, v) * numLayer, k or v: [batch, heads, seq_past, feature]
    public NDList pastKeyValuesList;

    // (k, v) * numLayer, k or v: [batch, heads, seq_past, feature]
    // Will be deprecated
    public NativeResource<Long> pastKeyValues;

    // Will be deprecated
    public CausalLMOutput(NDArray logits, NativeResource<Long> paskKeyValues) {
        this.logits = logits;
        this.pastKeyValues = paskKeyValues;
    }

    public CausalLMOutput(NDArray logits, NDList pastKeyValues) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValues;
    }
}
