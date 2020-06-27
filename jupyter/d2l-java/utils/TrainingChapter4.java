import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

public class TrainingChapter4{

    public static void sgd(NDList params, float lr, int batchSize) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr / batchSize
            param.subi(param.getGradient().mul(lr).div(batchSize));
        }
    }

    public static float accuracy(NDArray yHat, NDArray y) {
        // Check size of 1st dimension greater than 1
        // to see if we have multiple samples
        if (yHat.getShape().size(1) > 1) {
            // Argmax gets index of maximum args for given axis 1
            // Convert yHat to same dataType as y (int32)
            // Sum up number of true entries
            return yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                    .sum().toType(DataType.FLOAT32, false).getFloat();
        }
        return yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false))
                .sum().toType(DataType.FLOAT32, false).getFloat();
    }



}