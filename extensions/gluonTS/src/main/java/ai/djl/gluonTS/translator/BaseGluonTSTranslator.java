package ai.djl.gluonTS.translator;

import ai.djl.gluonTS.ForeCast;
import ai.djl.gluonTS.GluonTSData;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;

import java.util.Map;

/** Built-in {@code Translator} that provides default GluonTSTranslator config process. */
public abstract class BaseGluonTSTranslator implements Translator<GluonTSData, ForeCast> {

    protected int predictionLength;
    protected int context_length;

    protected String freq;

    private Batchifier batchifier;

    /**
     * Consturcts an GluonTSTranslator with the provied builder.
     *
     * @param builder the data to build with
     */
    public BaseGluonTSTranslator(BaseBuilder<?> builder) {
        this.batchifier = builder.batchifier;
        this.freq = builder.freq;
        this.predictionLength = builder.predictionLength;
        // TODO: for inferring
        this.context_length = builder.predictionLength;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /**
     * A builder to extend for all classes extend the {@link BaseGluonTSTranslator}.
     *
     * @param <T> the concrete builder type
     */
    public abstract static class BaseBuilder<T extends BaseBuilder<T>> {
        protected Batchifier batchifier = Batchifier.STACK;
        protected int predictionLength;

        protected String freq;

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier the {@link Batchifier} to be set
         * @return this builder
         */
        public T optBachifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return self();
        }

        protected abstract T self();

        protected void validate() {}

        protected void configPreProcess(Map<String, ?> arguments) {
            this.freq = ArgumentsUtil.stringValue(arguments, "freq", "D");
            this.predictionLength = ArgumentsUtil.intValue(arguments, "prediction_length");
            if (predictionLength <= 0) {
                throw new IllegalArgumentException(
                        "The value of `prediction_length` should be > 0");
            }
            if (arguments.containsKey("batchifier")) {
                batchifier = Batchifier.fromString((String) arguments.get("batchifier"));
            }
        }

        protected void configPostProcess(Map<String, ?> arguments) {}
    }
}
