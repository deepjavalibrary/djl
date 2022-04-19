package ai.djl.huggingface.tokenizers.jni;

public class CharSpan {
    private final int start;
    private final int end;

    public CharSpan(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public double getStart() {
        return start;
    }

    public double getEnd() {
        return end;
    }
}
