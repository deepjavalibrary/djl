package ai.djl.ndarray.internal;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDArrayExTest {

    @Test
    public void testToTensor() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            NDArray tensor = manager.create(127f).reshape(1, 1, 1, 1);
            Assert.assertNotNull(tensor.getNDArrayInternal().toTensor());
        }
    }

    @Test
    public void testToTensorMPS() {
        // Test that toTensor does not fail of MPS due to 32-bit support only
        if (System.getProperty("os.name").startsWith("Mac") &&
                System.getProperty("os.arch").equals("aarch64")) {
            try (NDManager manager = NDManager.newBaseManager(Device.fromName("mps"))) {
                NDArray tensor = manager.create(127f).reshape(1, 1, 1, 1);;
                Assert.assertNotNull(tensor.getNDArrayInternal().toTensor());
            }
        }
    }

}
