package ai.djl.audio.dataset;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Librispeech extends SpeechRecognitionDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "penntreebank-unlabeled-processed";

    /**
     * Creates a new instance of {@link SpeechRecognitionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public Librispeech(Builder builder) {
        super(builder);
        this.usage = builder.usage;
        this.mrl = builder.getMrl();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {
        if (prepared) {
            return;
        }
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Artifact.Item item;
        switch (usage) {
            case TRAIN:
                item = artifact.getFiles().get("train");
                break;
            case TEST:
                item = artifact.getFiles().get("test");
                break;
            default:
                throw new UnsupportedOperationException("Unsupported usage type.");
        }
        Path path = mrl.getRepository().getFile(item, "").toAbsolutePath();
        List<String> lineArray = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String row;
            while ((row = reader.readLine()) != null) {
                lineArray.add(row);
            }
        }
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        return null;
    }

    @Override
    protected long availableSize() {
        return 0;
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
