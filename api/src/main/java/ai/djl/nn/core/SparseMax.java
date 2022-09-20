package ai.djl.nn.core;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.stream.IntStream;

/**
 * {@code SparseMax} contains a generic implementation of sparsemax function
 * the definition of SparseMax can be referred to https://arxiv.org/pdf/1602.02068.pdf.
 * {@code SparseMax} is a simpler implementation of sparseMax function,
 * where we set K as a hyperParameter(default 3). We only do softmax on those max-K data,
 * and we set all the other value as 0.
 */
public class SparseMax extends AbstractBlock {
    private static final Byte VERSION = 1;

    private int axis;
    private int topK;
    private NDManager manager;

    public SparseMax(){
        this(-1,3);
    }

    public SparseMax(int axis){
        this(axis,3);
    }

    public SparseMax(int axis,int K){
        super(VERSION);
        this.axis = axis;
        this.topK = K;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        //the shape of input and output are the same
        return new Shape[0];
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        /*
        A simple implementation of sparseMax, where we only calculate softMax with largest K data
         */
        manager = inputs.getManager();
        NDArray input = inputs.singletonOrThrow();
        if(axis!=-1){
            input = input.swapAxes(axis,-1);
        }

        //level should be: the max i-th is index j in input
        NDArray level = input.argSort(-1,false).toType(DataType.INT64,false);
        int lastDimSize = (int)input.size(input.getShape().dimension()-1);

        //maskTopK should be: the topK in input is 1 and other is zero
        NDArray maskTopK = NDArrays.add(IntStream.range(0,topK).mapToObj(
                j-> level.get("..., {}",j).oneHot(lastDimSize)
        ).toArray(NDArray[]::new));

        NDArray expSum = input.exp().mul(maskTopK).sum(new int[]{-1},true).broadcast(input.getShape());
        NDArray output = input.exp().mul(maskTopK).div(expSum);

        if(axis!=-1) {
            output = output.swapAxes(axis, -1);
        }
        return new NDList(output);
    }
}
