package org.apache.mxnet.nn;

import com.amazon.ai.Block;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.PairList;
import java.util.Map;
import org.apache.mxnet.engine.MxNDFactory;
import org.apache.mxnet.jna.FunctionInfo;
import org.apache.mxnet.jna.JnaUtils;

public abstract class MxNNBlock implements Block {

    private static final Map<String, FunctionInfo> OPS = JnaUtils.getNdArrayFunctions();

    protected NDFactory factory;

    protected String opName;

    @Override
    public NDList forward(NDList inputs, Map<String, String> args) {
        PairList<String, String> params = new PairList<>(args);
        return OPS.get(opName).invoke((MxNDFactory) factory, inputs, params);
    }

    @Override
    public void backward() {}

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
