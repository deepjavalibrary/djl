package ai.djl.translate;

/** GPTConfig is used to store the GPT parameters used to select different versions of GPT. */
public class GPTConfig {
    public String[] modelUrls;
    public int numAttentionHeads;
    public int numLayers;
    public long hiddenStateDim;
    public long logitsDim;
    public long kvDim;

    public GPTConfig(String[] modelUrls) {
        this.modelUrls = modelUrls;
        numAttentionHeads = 12;
        numLayers = 12;
        hiddenStateDim = 768;
        logitsDim = 50257;
        kvDim = 64;
    }
}
