package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class PtNDArrayExTest {
    public static void main(String[] args) {
        PtNDArrayExTest ptNDArrayExTest = new PtNDArrayExTest();
        ptNDArrayExTest.testMultiBoxPrior();
    }

    @Test
    public void testMultiBoxPrior() {
        try (PtNDManager manager = (PtNDManager) NDManager.newBaseManager()) {
            Shape shape = new Shape(1, 64, 32, 32);
            DataType dataType = DataType.FLOAT32;
            PtNDArray ptNDArray = manager.create(shape, dataType);
            PtNDArrayEx ptNDArrayEx = new PtNDArrayEx(ptNDArray);

            List<Float> sizes = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            List<Float> ratios = new ArrayList<Float>(Arrays.asList(1f, 2f, 0.5f));
            List<Float> steps = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            List<Float> offsets = new ArrayList<Float>(Arrays.asList(0.2f, 0.272f));
            boolean clip = false;
            try (NDList result = ptNDArrayEx.multiBoxPrior(sizes, ratios, steps, offsets, clip)) {
                NDArray firstitem = result.get(0);
                Shape shapeResult = firstitem.getShape();
                assert(shapeResult.getLayout().length==2);
                assert(shapeResult.get(0)==6534);
                assert(shapeResult.get(1)==4);
                for (NDArray item: result) {
                    System.out.println(item);
                    //ND: (6534, 4) cpu() float32
                }
            }

            System.out.println("Done!");
        }
    }
}
