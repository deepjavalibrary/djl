package software.amazon.ai.integration.tests;

import java.util.stream.DoubleStream;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
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

    @RunAsTest
    public void testMax() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {2, 4, 6, 8}, null, new Shape(2, 2));

        Float maxAll = (Float) original.max();
        if (maxAll != 8) {
            throw new FailedTestException("Incorrect max all");
        }

        NDArray maxAxes = original.max(new int[] {1});
        NDArray maxAxesExpected = mxNDFactory.create(new float[] {4, 8}, null, new Shape(2));
        if (!maxAxes.contentEquals(maxAxesExpected)) {
            throw new FailedTestException("Incorrect max axes");
        }

        NDArray maxKeep = original.max(new int[] {0}, true);
        NDArray maxKeepExpected = mxNDFactory.create(new float[] {6, 8}, null, new Shape(1, 2));
        if (!maxKeep.contentEquals(maxKeepExpected)) {
            throw new FailedTestException("Incorrect max keep");
        }
    }

    @RunAsTest
    public void testMin() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {2, 4, 6, 8}, null, new Shape(2, 2));

        Float minAll = (Float) original.min();
        if (minAll != 2) {
            throw new FailedTestException("Incorrect min all");
        }

        NDArray minAxes = original.min(new int[] {1});
        NDArray minAxesExpected = mxNDFactory.create(new float[] {2, 6}, null, new Shape(2));
        if (!minAxes.contentEquals(minAxesExpected)) {
            throw new FailedTestException("Incorrect min axes");
        }

        NDArray minKeep = original.min(new int[] {0}, true);
        NDArray minKeepExpected = mxNDFactory.create(new float[] {2, 4}, null, new Shape(1, 2));
        if (!minKeep.contentEquals(minKeepExpected)) {
            throw new FailedTestException("Incorrect min keep");
        }
    }

    @RunAsTest
    public void testSum() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {2, 4, 6, 8}, null, new Shape(2, 2));

        Float sumAll = (Float) original.sum();
        if (sumAll != 20) {
            throw new FailedTestException("Incorrect sum all");
        }

        NDArray sumAxes = original.sum(new int[] {1});
        NDArray sumAxesExpected = mxNDFactory.create(new float[] {6, 14}, null, new Shape(2));
        if (!sumAxes.contentEquals(sumAxesExpected)) {
            throw new FailedTestException("Incorrect sum axes");
        }

        NDArray sumKeep = original.sum(new int[] {0}, true);
        NDArray sumKeepExpected = mxNDFactory.create(new float[] {8, 12}, null, new Shape(1, 2));
        if (!sumKeep.contentEquals(sumKeepExpected)) {
            throw new FailedTestException("Incorrect sum keep");
        }
    }

    @RunAsTest
    public void testProd() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {2, 4, 6, 8}, null, new Shape(2, 2));

        Float prodAll = (Float) original.prod();
        if (prodAll != 384) {
            throw new FailedTestException("Incorrect prod all");
        }

        NDArray prodAxes = original.prod(new int[] {1});
        NDArray prodAxesExpected = mxNDFactory.create(new float[] {8, 48}, null, new Shape(2));
        if (!prodAxes.contentEquals(prodAxesExpected)) {
            throw new FailedTestException("Incorrect prod axes");
        }

        NDArray prodKeep = original.prod(new int[] {0}, true);
        NDArray prodKeepExpected = mxNDFactory.create(new float[] {12, 32}, null, new Shape(1, 2));
        if (!prodKeep.contentEquals(prodKeepExpected)) {
            throw new FailedTestException("Incorrect prod keep");
        }
    }

    @RunAsTest
    public void testMean() throws FailedTestException {
        NDArray original = mxNDFactory.create(new float[] {2, 4, 6, 8}, null, new Shape(2, 2));

        Float meanAll = (Float) original.mean();
        if (meanAll != 5) {
            throw new FailedTestException("Incorrect mean all");
        }

        NDArray meanAxes = original.mean(new int[] {1});
        NDArray meanAxesExpected = mxNDFactory.create(new float[] {3, 7}, null, new Shape(2));
        if (!meanAxes.contentEquals(meanAxesExpected)) {
            throw new FailedTestException("Incorrect mean axes");
        }

        NDArray meanKeep = original.mean(new int[] {0}, true);
        NDArray meanKeepExpected = mxNDFactory.create(new float[] {4, 6}, null, new Shape(1, 2));
        if (!meanKeep.contentEquals(meanKeepExpected)) {
            throw new FailedTestException("Incorrect mean keep");
        }
    }

    @RunAsTest
    public void testLogicalNot() throws FailedTestException {
        double[] testedData = new double[] {-2., 0., 1.};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        double[] boolData = new double[] {0.0, 1.0, 0.0};
        MxNDArray expectedND = mxNDFactory.create(boolData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.logicalNot(), expectedND);
    }

    @RunAsTest
    public void testAbs() throws FailedTestException {
        double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::abs).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.abs(), expectedND);
    }

    @RunAsTest
    public void testSquare() throws FailedTestException {
        double[] testedData = new double[] {1.0, -2.12312, -3.5784, -4.0, 5.0, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(x -> Math.pow(x, 2.0)).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.square(), expectedND);
    }

    @RunAsTest
    public void testCbrt() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::cbrt).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.cbrt(), expectedND);
    }

    @RunAsTest
    public void testFloor() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::floor).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.floor(), expectedND);
    }

    @RunAsTest
    public void testCeil() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::ceil).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.ceil(), expectedND);
    }

    @RunAsTest
    public void testRound() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::round).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.round(), expectedND);
    }

    @RunAsTest
    public void testTrunc() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        double[] truncData = new double[] {1.0, 2.0, -3, -4, 5, -223};
        MxNDArray expectedND = mxNDFactory.create(truncData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.trunc(), expectedND);
    }

    @RunAsTest
    public void testExp() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, -3.584, -4.343234, 5.11111, -223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::exp).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.exp(), expectedND);
    }

    @RunAsTest
    public void testLog() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::log).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.log(), expectedND);
    }

    @RunAsTest
    public void testLog10() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::log10).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.log10(), expectedND);
    }

    @RunAsTest
    public void testLog2() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(x -> Math.log10(x) / Math.log10(2)).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.log2(), expectedND);
    }

    @RunAsTest
    public void testSin() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::sin).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.sin(), expectedND);
    }

    @RunAsTest
    public void testCos() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::cos).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.cos(), expectedND);
    }

    @RunAsTest
    public void testTan() throws FailedTestException {
        double[] testedData = new double[] {0.0, Math.PI / 4.0, Math.PI / 2.0};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::tan).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.tan(), expectedND);
    }

    @RunAsTest
    public void testAsin() throws FailedTestException {
        double[] testedData = new double[] {1.0, -1.0, -0.22, 0.4, 0.1234};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::asin).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.asin(), expectedND);
    }

    @RunAsTest
    public void testAcos() throws FailedTestException {
        double[] testedData = new double[] {-1.0, -0.707, 0.0, 0.707, 1.0};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::acos).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.acos(), expectedND);
    }

    @RunAsTest
    public void testAtan() throws FailedTestException {
        double[] testedData = new double[] {-1.0, 0.0, 1.0};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::atan).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.atan(), expectedND);
    }

    @RunAsTest
    public void testToDegrees() throws FailedTestException {
        double[] testedData = new double[] {0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::toDegrees).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.toDegrees(), expectedND);
    }

    @RunAsTest
    public void testToRadians() throws FailedTestException {
        double[] testedData = new double[] {0.0, 90.0, 180.0, 270.0, 360.0};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::toRadians).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.toRadians(), expectedND);
    }

    @RunAsTest
    public void testSinh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::sinh).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.sinh(), expectedND);
    }

    @RunAsTest
    public void testCosh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::cosh).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.cosh(), expectedND);
    }

    @RunAsTest
    public void testTanh() throws FailedTestException {
        double[] testedData = new double[] {1.0, 2.2312, 3.584, 4.343234, 5.11111, 223.23423};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        testedData = DoubleStream.of(testedData).map(Math::tanh).toArray();
        MxNDArray expectedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.tanh(), expectedND);
    }

    @RunAsTest
    public void testAsinh() throws FailedTestException {
        double[] testedData = new double[] {Math.E, 10.0};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        double[] aSinhData = new double[] {1.72538256, 2.99822295};
        MxNDArray expectedND = mxNDFactory.create(aSinhData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.asinh(), expectedND);
    }

    @RunAsTest
    public void testAtanh() throws FailedTestException {
        double[] testedData = new double[] {0.0, -0.5};
        MxNDArray testedND = mxNDFactory.create(testedData, null, new Shape(testedData.length));
        double[] aTanhData = new double[] {0.0, -0.54930614};
        MxNDArray expectedND = mxNDFactory.create(aTanhData, null, new Shape(testedData.length));
        Assertions.assertAlmostEquals(testedND.atanh(), expectedND);
    }
}
