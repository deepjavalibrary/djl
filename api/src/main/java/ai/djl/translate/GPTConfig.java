package ai.djl.translate;

/** GPTConfig is used to store the GPT parameters used to select different versions of GPT. */
public class GPTConfig {
    public long hiddenSize;
    public long numAttentionHeads;

    public int numLayers;

    public GPTConfig() {
        hiddenSize = 64;
        numAttentionHeads = 12;
        numLayers = 12;
    }
}
