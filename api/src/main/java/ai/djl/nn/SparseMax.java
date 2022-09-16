package ai.djl.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * {@code SparseMax} contains a generic implementation of sparsemax function
 * the definition of SparseMax can be referred to https://arxiv.org/pdf/1602.02068.pdf
 * this implementation is a simpler implementation of sparseMax function,
 * where we set K as a hyperParameter, and we only do softmax on those max-K data
 * and we set all the other value as 0.
 */
public class SparseMax extends AbstractBlock{
    private static final Byte VERSION = 1;

    private int axis;
    private int K;
    private NDManager manager;

    public SparseMax(){
        this(-1,3);
    }

    public SparseMax(int axis,int K){
        super(VERSION);
        this.axis = axis;
        this.K = K;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        //the shape of input and output are the same
        return new Shape[0];
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        System.out.println("In sparseMax forwardInternal");
        manager = inputs.getManager();
        NDArray input = inputs.singletonOrThrow();
        if(this.axis!=-1){
            input = input.swapAxes(this.axis,-1);   //swap the axes and get what we want to the last
        }

        //have a problem here with argSort
        System.out.println("input = "+input);
        NDArray mask = input.argSort(1,true);
        System.out.println("mask = "+mask);

        mask = NDArrays.where(mask.lt(K),manager.ones(mask.getShape()),manager.zeros(mask.getShape()));
        System.out.println("mask = "+mask);

        NDArray sum = mask.mul(input.exp()).sum(new int[]{-1},true).broadcast(mask.getShape());
        System.out.println("sum = "+sum);

        NDArray result = mask.mul(input.exp().div(sum));
        System.out.println("result = "+result);
        return new NDList(result);
    }
}
