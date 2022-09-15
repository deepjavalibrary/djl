package ai.djl.integration.tests.model_zoo.tabular;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.SparseMax;
import ai.djl.training.ParameterStore;

public class TabNetTest {
    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NDArray array = manager.randomUniform(0,1,new Shape(2,4), DataType.FLOAT32);
        System.out.println(array);
        System.out.println(array.argSort(-1,false));
        SparseMax sparseMax = new SparseMax();
        ParameterStore parameterStore = new ParameterStore();
        sparseMax.forward(parameterStore,new NDList(array),true);
    }
}
