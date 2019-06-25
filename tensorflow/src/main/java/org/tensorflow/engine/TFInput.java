package org.tensorflow.engine;


public interface TFInput {
    public String setOutputName();

    public <I> I getInput();
}
