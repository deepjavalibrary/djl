package org.apache.mxnet.nn;

import com.amazon.ai.Block;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.PairList;

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
