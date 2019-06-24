package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.Profiler;
import com.amazon.ai.Translator;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.EngineUtils;
import com.amazon.ai.nn.NNIndex;
import com.amazon.ai.training.Trainer;
import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import org.tensorflow.TensorFlow;

public class TfEngine extends Engine {

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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return TensorFlow.version();
    }

    /** {@inheritDoc} */
    @Override
    public Model loadModel(Path modelPath, String modelName, int epoch) throws IOException {
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
    public EngineUtils getEngineUtils() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(Model model, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setProfiler(Profiler profiler) {}
}
