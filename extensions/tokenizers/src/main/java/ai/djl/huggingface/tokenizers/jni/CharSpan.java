package ai.djl.huggingface.tokenizers.jni;


public class CharSpan {
    private double start;
    private double end;

    public CharSpan(int start, int end) {
        this.start = start;
        this.end = end;
    }

}