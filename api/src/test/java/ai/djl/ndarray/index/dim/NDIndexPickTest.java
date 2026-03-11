package ai.djl.ndarray.index.dim;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDIndexPickTest {

    @Test
    public void testPickDim() {
        for (String engine: new String[]{"MXNet", "PyTorch"}) {
            try (NDManager manager = NDManager.newBaseManager(engine)) {
                NDArray picker = manager.zeros(new Shape(1, 5));
                picker.set(new NDIndex(0, 0), 1);
                picker.set(new NDIndex(0, 2), 1);
                picker.set(new NDIndex(0, 4), 1);
                NDArray values = manager.create(
                                new float[] {
                                        0f, 0.1f, 0.08f, 0.52f, 0.92f, 1f, 0.55f, 0.2f, 0.9f, 0.88f
                                })
                        .reshape(new Shape(1, 5, 2));
                NDIndex pickIndex =
                        new NDIndex()
                                .addAllDim(2)
                                .addPickDim(picker);
                NDArray out = values.get(pickIndex);
                for (int i = 0; i < out.size(1); i++) {
                    Assert.assertEquals(
                            out.getFloat(0, i, 0),
                            values.getFloat(0, i, (long) picker.getFloat(0, i)));
                }
            }
        }
    }

}
