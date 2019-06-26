package org.apache.mxnet.nn;

import software.amazon.ai.Block;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.PairList;

public abstract class MxNNBlock implements Block {

    protected String opName;

    @Override
    public NDList forward(NDList inputs, PairList<String, String> params) {
        NDArray[] inputArray = inputs.toArray();
        NDFactory factory = inputArray[0].getFactory();
        NDArray[] output = factory.invoke(opName, inputs.toArray(), null, params);
        return new NDList(output);
    }

    @Override
    public void backward() {}

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
