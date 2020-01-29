package ai.djl.pytorch;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;
import java.util.Arrays;

public class PytorchTest {
    public static void main(String[] args) {
        testTensor();
        testModule();
    }

    public static void testPrintTensor(Pointer tensorHandle) {
        PointerByReference resultHandleRef = new PointerByReference();
        PointerByReference shapeRef = new PointerByReference();
        IntBuffer buf2 = IntBuffer.allocate(1);
        IntBuffer buf3 = IntBuffer.allocate(1);
        PyTorchLibrary.INSTANCE.TensorToFloat(tensorHandle, resultHandleRef, buf2);
        PyTorchLibrary.INSTANCE.TensorGetShape(tensorHandle, buf3, shapeRef);
        float[] data = resultHandleRef.getValue().getFloatArray(0, buf2.get());
        long[] shape = shapeRef.getValue().getLongArray(0, buf3.get());
        System.out.println(Arrays.toString(data));
        System.out.println(Arrays.toString(shape));
    }

    public static void testTensor() {
        PointerByReference ref = new PointerByReference();
        PyTorchLibrary.INSTANCE.ones(new long[] {1, 3, 224, 224}, 4, ref);
        testPrintTensor(ref.getValue());
    }

    public static void testModule() {
        PointerByReference tensorHandleRef = new PointerByReference();
        PyTorchLibrary.INSTANCE.ones(new long[] {1, 3, 224, 224}, 4, tensorHandleRef);
        Pointer inputTensorHandle = tensorHandleRef.getValue();
        PointerByReference moduleHandleRef = new PointerByReference();
        PyTorchLibrary.INSTANCE.ModuleLoad(System.getProperty("user.dir") + "/pytorch/pytorch-engine/traced_resnet_model.pt", moduleHandleRef);
        PyTorchLibrary.INSTANCE.ModuleEval(moduleHandleRef.getValue());
        PointerByReference iValueHandleRef = new PointerByReference();
        PyTorchLibrary.INSTANCE.IValueCreateFromTensor(tensorHandleRef.getValue(), iValueHandleRef);
        PointerByReference resultHandleRef = new PointerByReference();
        PyTorchLibrary.INSTANCE.ModuleForward(moduleHandleRef.getValue(), iValueHandleRef.getPointer(), 1, resultHandleRef);
        PointerByReference resultTensorHandleRef = new PointerByReference();
        PyTorchLibrary.INSTANCE.IValueToTensor(resultHandleRef.getValue(), resultTensorHandleRef);
        testPrintTensor(resultTensorHandleRef.getValue());
    }
}
