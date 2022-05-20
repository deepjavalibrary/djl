package ai.djl.nn.norm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class GhostBatchNormTest {

    private GhostBatchNorm gbn;
    private NDManager manager;

    @BeforeTest
    void initialize() {
        this.manager = NDManager.newBaseManager();
    }

    private void testSubBatchingShapeSize(Shape shape, int vbs, int expectedSize) {
        NDList[] zeroSubList = generateZerosArrayAndSubBatch(shape, vbs);
        Assert.assertEquals(expectedSize, zeroSubList.length);
    }

    private NDList[] generateZerosArrayAndSubBatch(Shape shape, int vbs) {
        this.gbn = new GhostBatchNorm(GhostBatchNorm.builder().optVirtualBatchSize(vbs));
        NDArray zerosInput = manager.zeros(shape);
        return gbn.split(new NDList(zerosInput));
    }

    @Test
    public void originalBatchOfSizeOne_subBatchIntoOneVBS_listOfLengthOne() {
        testSubBatchingShapeSize(new Shape(1, 1, 10), 1, 1);
    }

    @Test
    public void originalBatchOfSizeSix_subBatchIntoTwoVBS_listOfLengthThree() {
        testSubBatchingShapeSize(new Shape(6, 1, 10), 2, 3);
    }

    @Test
    public void originalBatchOfSize60_subBatchInto64VBS_listOfLengthOne() {
        testSubBatchingShapeSize(new Shape(60, 10), 64, 1);
    }
}
