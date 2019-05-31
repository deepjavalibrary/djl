package com.amazon.ai.nn.core;

import com.amazon.ai.Block;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;

public interface Linear extends Block {

    NDArray forward(NDArray data);

    static Linear create(int units, int inUnits, NDFactory factory) {
        return Engine.getInstance().getNNIndex().linear(units, inUnits, factory);
    }
}
