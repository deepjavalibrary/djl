package ai.djl.util;

public interface NativeResource<T> extends AutoCloseable {
    boolean isReleased();

    T getHandle();

    String getUid();

    @Override
    void close();
}
