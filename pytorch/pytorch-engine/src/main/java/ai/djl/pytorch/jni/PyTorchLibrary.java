package ai.djl.pytorch.jni;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class PyTorchLibrary {
    public native Pointer atOnes(long[] shape);

    public native long[] atSizes(Pointer handle);

    public native ByteBuffer atDataPtr(Pointer handle);

    public static void main(String[] args) {
        System.loadLibrary("djl_torch");
        PyTorchLibrary library = new PyTorchLibrary();
        Pointer pointer = library.atOnes(new long[]{2, 3, 4});
        long[] sizes = library.atSizes(pointer);
        System.out.println(Arrays.toString(sizes));
        ByteBuffer bb = library.atDataPtr(pointer);
        bb = bb.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer fb = bb.asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        System.out.println(Arrays.toString(ret));
    }
}
