package ai.djl.pytorch;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;

public interface PyTorchLibrary extends Library {

    PyTorchLibrary INSTANCE =
            Native.load(
                    System.getProperty("user.dir")
                            + "/pytorch/pytorch-engine/c_api/build/libtorch-wrapper.dylib",
                    PyTorchLibrary.class);

    int ones(PointerByReference output);

    int TensorToFloat(Pointer input, PointerByReference output, IntBuffer size);

    int TensorGetShape(Pointer input, IntBuffer dim, PointerByReference output);
}
