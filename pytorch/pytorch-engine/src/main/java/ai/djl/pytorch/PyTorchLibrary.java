package ai.djl.pytorch;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.CharBuffer;
import java.nio.IntBuffer;

public interface PyTorchLibrary extends Library {

    PyTorchLibrary INSTANCE =
            Native.load(
                    System.getProperty("user.dir")
                            + "/pytorch/pytorch-engine/c_api/build/libtorch-wrapper.dylib",
                    PyTorchLibrary.class);

    int ones(PointerByReference output);

    int TensorToFloat(Pointer tensorHandle, PointerByReference output, IntBuffer size);

    int TensorGetShape(Pointer tensorHandle, IntBuffer dim, PointerByReference output);

    int ModuleLoad(String path, PointerByReference moduleHandle);

    int ModuleEval(Pointer moduleHandle);

    int ModuleForward(Pointer moduleHandle, Pointer iValueArrayHandle, int length, PointerByReference resultHandle);

    int IValueCreateFromTensor(Pointer tensorHandle, PointerByReference iValueHandle);

    int IValueToTensor(Pointer iValueHandle, PointerByReference tensorHandle);
}
