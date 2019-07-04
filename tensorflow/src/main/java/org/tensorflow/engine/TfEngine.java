package org.tensorflow.engine;

import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import java.util.Map;
import org.tensorflow.TensorFlow;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.Translator;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.nn.NNIndex;
import software.amazon.ai.training.Trainer;

public class TfEngine extends Engine {

    TfEngine() {}

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "Tensorflow";
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public MemoryUsage getGpuMemory(Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Context defaultContext() {
        return Context.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return TensorFlow.version();
    }

    /** {@inheritDoc} */
    @Override
    public Model loadModel(Path modelPath, String modelName, Map<String, String> options) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(
            Model model, Translator<I, O> translator, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NNIndex getNNIndex() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(Model model, Context context) {
        return null;
    }
}
