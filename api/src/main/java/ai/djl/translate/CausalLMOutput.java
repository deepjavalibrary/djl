package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** CausalLMOuput is used to contain multiple output of a language model. */
public class CausalLMOutput {

    // [batch, seq, feature]
    // The prob. conditional on a sequence that ends at an element in seq-dim. seq-dim-size =
    // |inputIds|
    public NDArray logits;

    // [batch, seq, dim] * (layers+1) -> take -1
    // The vec. rep. of a sequence that ends at an element in seq-dim. seq-dim-size = |inputIds|
    public NDList allHiddenStates;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, feature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|
    public NDList pastKeyValuesList;

    public CausalLMOutput(NDArray logits, NDList pastKeyValues) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValues;
    }

    public CausalLMOutput(NDArray logits, NDList... optionalOutput) {
        this.logits = logits;
        this.pastKeyValuesList = optionalOutput[0];
        this.allHiddenStates = optionalOutput[1];
    }
}
