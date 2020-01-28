package ai.djl.pytorch;

import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;

public class PytorchTest {
    public static void main(String[] args) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = new PointerByReference();
        System.out.println(PyTorchLibrary.INSTANCE.getShape(null, dim, ref));
    }
}
