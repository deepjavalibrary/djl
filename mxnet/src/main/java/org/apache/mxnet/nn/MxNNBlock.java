package org.apache.mxnet.nn;

import com.amazon.ai.Block;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFuncParams;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.PairList;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDFactory;
import org.apache.mxnet.jna.JnaUtils;

public abstract class MxNNBlock implements Block {

    protected String opName;

    @Override
    public NDList forward(NDList inputs, PairList<String, String> params, NDFuncParams fparams) {
        NDArray[] inputArray = inputs.toArray();
        MxNDArray[] output =
                JnaUtils.op(opName)
                        .invoke(
                                (MxNDFactory) inputs.get(0).getFactory(),
                                (MxNDArray[]) inputArray,
                                params,
                                fparams);
        return new NDList(output);
    }

    @Override
    public void backward() {}

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
