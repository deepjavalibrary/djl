package ai.djl.audio.dataset;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.nlp.PennTreebankText;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Dataset;

public class Librispeech extends SpeechRecognitionDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "penntreebank-unlabeled-processed";

    /**
     * Creates a new instance of {@link SpeechRecognitionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public Librispeech(AudioBuilder<?> builder) {
        super(builder);
    }





    /** A builder to construct a {@link Librispeech} . */
    public static class Builder extends AudioBuilder<Librispeech.Builder> {

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Dataset.Usage.TRAIN;
        }

        /**
         * Builds a new {@link Librispeech} object.
         *
         * @return the new {@link Librispeech} object
         */
        public Librispeech build() {
            return new Librispeech(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.NLP.ANY, groupId, artifactId, VERSION);
        }

        /** {@inheritDoc} */
        @Override
        protected Librispeech.Builder self() {
            return this;
        }
    }

}
