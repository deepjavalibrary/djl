package ai.djl.pytorch;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.IntBuffer;

public interface PyTorchLibrary extends Library {

    PyTorchLibrary INSTANCE = (PyTorchLibrary) Native.load("/Volumes/unix/Joule/pytorch/pytorch-engine/c_api/build/libtorch-wrapper.dylib", PyTorchLibrary.class);

    int getShape(Pointer input, IntBuffer dim, PointerByReference output);
}
