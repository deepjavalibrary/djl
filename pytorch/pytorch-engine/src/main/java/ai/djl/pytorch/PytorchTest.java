package ai.djl.pytorch;

import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;
import java.util.Arrays;

public class PytorchTest {
    public static void main(String[] args) {
        IntBuffer buf = IntBuffer.allocate(1);
        IntBuffer buf2 = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        PointerByReference ref1 = new PointerByReference();
        PyTorchLibrary.INSTANCE.ones(ref);
        PyTorchLibrary.INSTANCE.TensorToFloat(ref.getValue(), ref1, buf2);
        float[] data = ref1.getValue().getFloatArray(0, buf2.get());
        System.out.println(Arrays.toString(data));
    }
}
