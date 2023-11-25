package ai.djl.modality;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class ClassificationsTest {

    @Test
    public void testClassificationsMPS() {
        // Test that toTensor does not fail of MPS due to 32-bit support only
        if (System.getProperty("os.name").startsWith("Mac") &&
                System.getProperty("os.arch").equals("aarch64")) {
            try (NDManager manager = NDManager.newBaseManager(Device.fromName("mps"))) {
                List<String> names = Arrays.asList("First", "Second", "Third", "Fourth", "Fifth");
                NDArray tensor = manager.create(new float[]{0f, 0.125f, 1f, 0.5f, 0.25f});
                Classifications classifications = new Classifications(
                        names,
                        tensor
                );
                Assert.assertNotNull(classifications.topK(1).equals(Arrays.asList("Third")));
            }
        }
    }

}
