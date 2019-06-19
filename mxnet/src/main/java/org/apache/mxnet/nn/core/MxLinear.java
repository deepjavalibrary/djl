package org.apache.mxnet.nn.core;

import com.amazon.ai.Initializer;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDFuncParams;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.nn.core.Linear;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;

public class MxLinear extends MxNNBlock implements Linear {

    private NDArray weight;
    private NDArray bias;

    private int units;
    private int inUnits;

    public MxLinear(int units, int inUnits) {
        this.opName = "FullyConnected";
        this.units = units;
        this.inUnits = inUnits;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getInputShape() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<NDArray> getDirectParameters() {
        return Arrays.asList(weight, bias);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDFactory factory, Initializer initializer) {
        weight = factory.create(new DataDesc(new Shape(units, inUnits)));
        bias = factory.create(new DataDesc(new Shape(units)));
        initializer.initialize(getDirectParameters());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data, NDFuncParams fparams) {
        NDList inputs = new NDList(data, weight, bias);
        MxOpParams params = new MxOpParams();
        params.add("num_hidden", "1");
        return forward(inputs, params, fparams).get(0);
    }
}
