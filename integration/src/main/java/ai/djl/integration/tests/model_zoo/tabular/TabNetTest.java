package ai.djl.integration.tests.model_zoo.tabular;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;

public class TabNetTest {
    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NDArray array = manager.create(new float[]{1,2,3,4});
        array = array.reshape(new Shape(2,2));
        System.out.println(array);
        NDArray res = Activation.tabNetGLU(array,1);
        System.out.println(res);
    }
}
