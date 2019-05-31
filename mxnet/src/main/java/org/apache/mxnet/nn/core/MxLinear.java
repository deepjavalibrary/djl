package org.apache.mxnet.nn.core;

import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.nn.core.Linear;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.nn.MxNNBlock;

public class MxLinear extends MxNNBlock implements Linear {

    private NDArray weight;

    private NDArray bias;

    public MxLinear(int units, int inUnits, NDFactory factory) {
        this.opName = "FullyConnected";
        this.factory = factory;
        this.weight = factory.create(new DataDesc(new Shape(units, inUnits)));
        this.bias = factory.create(new DataDesc(new Shape(units)));
    }

    @Override
    public Shape getInputShape() {
        return null;
    }

    @Override
    public List<NDArray> getParameters() {
        return Arrays.asList(weight, bias);
    }

    public NDArray forward(NDArray data) {
        NDList inputs = new NDList(data, weight, bias);
        Map<String, String> params = new HashMap<>();
        params.put("num_hidden", "1");
        return forward(inputs, params).get(0);
    }
}
