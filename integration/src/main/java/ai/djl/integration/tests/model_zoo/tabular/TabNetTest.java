package ai.djl.integration.tests.model_zoo.tabular;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.SparseMax;
import ai.djl.training.ParameterStore;

public class TabNetTest {
    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        //System.out.println(IntStream.range(0,3));
        NDArray array = manager.randomUniform(0,1,new Shape(4,2), DataType.FLOAT32);
        //System.out.println(array);
        SparseMax sparseMax = new SparseMax(0);
        ParameterStore parameterStore = new ParameterStore();
        NDArray output = sparseMax.forward(parameterStore,new NDList(array),true).singletonOrThrow();
        System.out.println(output);
    }
}
