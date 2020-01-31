package ai.djl.pytorch.jni;

public class Pointer {
    protected long peer;
    public Pointer(long peer) {
        this.peer = peer;
    }
    long getValue() {
        return peer;
    }
}
