package software.amazon.ai.integration.tests;

import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;

public class MxNDArrayOperatorsTest extends AbstractTest {

    MxNDFactory mxNDFactory = MxNDFactory.getSystemFactory();

    public static void main(String[] args) {
        new MxNDArrayOperatorsTest().runTest(args);
    }

    @RunAsTest
    public void testCopyTo() throws FailedTestException {
        MxNDArray mxNDArray1 =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray2 = mxNDFactory.create(new DataDesc(new Shape(1, 4)));
        mxNDArray1.copyTo(mxNDArray2);
        if (mxNDArray1.eq(mxNDArray2).nonzero() != mxNDArray1.size()) {
            throw new FailedTestException("CopyTo NDArray failed");
        }
    }

    @RunAsTest
    public void testEqualsForEqualNDArray() throws FailedTestException {
        MxNDArray mxNDArray1 =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray2 =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        NDArray result = mxNDArray1.eq(mxNDArray2);
        if (result.nonzero() != 4 || !mxNDArray1.equals(mxNDArray2)) {
            throw new FailedTestException("Incorrect comparison for equal NDArray");
        }
    }

    @RunAsTest
    public void testEqualsForScalar() throws FailedTestException {
        MxNDArray mxNDArray =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        if (mxNDArray.eq(2).nonzero() != 1) {
            throw new FailedTestException("Incorrect comparison for equal NDArray");
        }
    }

    @RunAsTest
    public void testEqualsForUnEqualNDArray() throws FailedTestException {
        MxNDArray mxNDArray1 =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray2 =
                mxNDFactory.create(new float[] {1f, 3f, 3f, 4f}, null, new Shape(1, 4));
        NDArray result = mxNDArray1.eq(mxNDArray2);
        if (result.nonzero() != 3 || mxNDArray1.equals(mxNDArray2)) {
            throw new FailedTestException("Incorrect comparison for unequal NDArray");
        }
    }

    @RunAsTest
    public void testNonZero() throws FailedTestException {
        MxNDArray mxNDArray1 =
                mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray2 =
                mxNDFactory.create(new float[] {1f, 2f, 0f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray3 =
                mxNDFactory.create(new float[] {0f, 0f, 0f, 4f}, null, new Shape(1, 4));
        MxNDArray mxNDArray4 =
                mxNDFactory.create(new float[] {0f, 0f, 0f, 0f}, null, new Shape(1, 4));
        if (mxNDArray1.nonzero() != 4
                || mxNDArray2.nonzero() != 3
                || mxNDArray3.nonzero() != 1
                || mxNDArray4.nonzero() != 0) {
            throw new FailedTestException("nonzero function returned incorrect value");
        }
    }

    @RunAsTest
    public void testAddScalar() throws FailedTestException {
        MxNDArray addend = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray summedArray = (MxNDArray) addend.add(2f);
        if (summedArray.getHandle().equals(addend.getHandle())) {
            throw new FailedTestException("Unexpected in-place summation");
        }
        MxNDArray result = mxNDFactory.create(new float[] {3f, 4f, 5f, 6f}, null, new Shape(1, 4));
        if (summedArray.eq(result).nonzero() != addend.size()) {
            throw new FailedTestException("Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testAddScalarInPlace() throws FailedTestException {
        MxNDArray addend = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray summedArray = (MxNDArray) addend.addi(2f);
        if (!summedArray.getHandle().equals(addend.getHandle())) {
            throw new FailedTestException("In-place summation failed");
        }
        MxNDArray result = mxNDFactory.create(new float[] {3f, 4f, 5f, 6f}, null, new Shape(1, 4));
        if (summedArray.eq(result).nonzero() != addend.size()) {
            throw new FailedTestException("Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testAddNDArray() throws FailedTestException {
        MxNDArray addend = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray addendum =
                mxNDFactory.create(new float[] {2f, 3f, 4f, 5f}, null, new Shape(1, 4));
        MxNDArray summedArray = (MxNDArray) addend.add(addendum);
        if (summedArray.getHandle().equals(addend.getHandle())) {
            throw new FailedTestException("Unexpected in-place summation");
        }
        MxNDArray solution =
                mxNDFactory.create(new float[] {3f, 5f, 7f, 9f}, null, new Shape(1, 4));
        if (solution.eq(summedArray).nonzero() != addend.size()) {
            throw new FailedTestException("Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testAddNDArrayInPlace() throws FailedTestException {
        MxNDArray addend = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(1, 4));
        MxNDArray addendum =
                mxNDFactory.create(new float[] {2f, 3f, 4f, 5f}, null, new Shape(1, 4));

        MxNDArray summedArray = (MxNDArray) addend.addi(addendum);
        if (!summedArray.getHandle().equals(addend.getHandle())) {
            throw new FailedTestException("Unexpected in-place summation");
        }
        MxNDArray solution =
                mxNDFactory.create(new float[] {3f, 5f, 7f, 9f}, null, new Shape(1, 4));
        if (solution.eq(summedArray).nonzero() != addend.size()) {
            throw new FailedTestException("Incorrect value in summed array");
        }
    }

    @RunAsTest
    public void testTile() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(2, 2));

        NDArray tileAll = original.tile(2);
        NDArray tileAllExpected =
                mxNDFactory.create(
                        new float[] {1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4},
                        null,
                        new Shape(4, 4));
        if (!tileAll.contentEquals(tileAllExpected)) {
            throw new FailedTestException("Incorrect tile all");
        }

        NDArray tileAxis = original.tile(0, 3);
        NDArray tileAxisExpected =
                mxNDFactory.create(
                        new float[] {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, null, new Shape(6, 2));
        if (!tileAxis.contentEquals(tileAxisExpected)) {
            throw new FailedTestException("Incorrect tile on axis");
        }

        NDArray tileArray = original.tile(new int[] {3, 1});
        if (!tileArray.contentEquals(tileAxisExpected)) {
            throw new FailedTestException("Incorrect tile array");
        }

        NDArray tileShape = original.tile(new Shape(4));
        NDArray tileShapeExpected =
                mxNDFactory.create(new float[] {1, 2, 1, 2, 3, 4, 3, 4}, null, new Shape(2, 4));
        if (!tileShape.contentEquals(tileShapeExpected)) {
            throw new FailedTestException("Incorrect tile shape");
        }
    }

    @RunAsTest
    public void testRepeat() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {1f, 2f, 3f, 4f}, null, new Shape(2, 2));

        NDArray repeatAll = original.repeat(2);
        NDArray repeatAllExpected =
                mxNDFactory.create(
                        new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4},
                        null,
                        new Shape(4, 4));
        if (!repeatAll.contentEquals(repeatAllExpected)) {
            throw new FailedTestException("Incorrect repeat all");
        }

        NDArray repeatAxis = original.repeat(0, 3);
        NDArray repeatAxisExpected =
                mxNDFactory.create(
                        new float[] {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}, null, new Shape(6, 2));
        if (!repeatAxis.contentEquals(repeatAxisExpected)) {
            throw new FailedTestException("Incorrect repeat on axis");
        }

        NDArray repeatArray = original.repeat(new int[] {3, 1});
        if (!repeatArray.contentEquals(repeatAxisExpected)) {
            throw new FailedTestException("Incorrect repeat array");
        }

        NDArray repeatShape = original.repeat(new Shape(4));
        NDArray repeatShapeExpected =
                mxNDFactory.create(new float[] {1, 1, 2, 2, 3, 3, 4, 4}, null, new Shape(2, 4));
        if (!repeatShape.contentEquals(repeatShapeExpected)) {
            throw new FailedTestException("Incorrect repeat shape");
        }
    }
}
