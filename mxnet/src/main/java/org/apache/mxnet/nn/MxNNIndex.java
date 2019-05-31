package org.apache.mxnet.nn;

import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.nn.core.Linear;
import org.apache.mxnet.nn.core.MxLinear;

public class MxNNIndex extends NNIndex {
    public Linear linear(int units, int inUnits, NDFactory factory) {
        return new MxLinear(units, inUnits, factory);
    }
}
