package ai.djl.mxnet.engine;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MxNDArrayExTest {
    public static void main(String[] args) {
        MxNDArrayExTest mxNDArrayExTest = new MxNDArrayExTest();
        mxNDArrayExTest.testMultiBoxPrior();
    }

    @Test
    public void testMultiBoxPrior() {
        Engine engine = Engine.getInstance();

        try (NDManager manager =  NDManager.newBaseManager()) {

            Shape shape = new Shape(1, 64, 32, 32);
            DataType dataType = DataType.FLOAT32;
            NDArray mxNDArray = manager.create(shape, dataType);
            NDArrayEx mxNDArrayEx = mxNDArray.getNDArrayInternal();

            List<Float> sizes = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            List<Float> ratios = new ArrayList<Float>(Arrays.asList(1f, 2f, 0.5f));
            List<Float> steps = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            List<Float> offsets = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            boolean clip = false;
            try (NDList result = mxNDArrayEx.multiBoxPrior(sizes, ratios, steps, offsets, clip)) {
                NDArray firstitem = result.get(0);
                Shape shapeResult = firstitem.getShape();

                assert(shapeResult.getLayout().length==2);
                assert(shapeResult.get(0)==6534);
                assert(shapeResult.get(1)==4);
                for (NDArray item: result) {
                    System.out.println(item);
                    //ND: (1, 4096, 4) cpu() float32
                }

            }
            System.out.println("Done!");
        }
    }
}

