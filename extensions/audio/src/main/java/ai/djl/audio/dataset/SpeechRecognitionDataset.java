package ai.djl.audio.dataset;

import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.List;

public abstract class SpeechRecognitionDataset extends RandomAccessDataset {

    protected AudioData sourceAudioData;
    protected TextData targetTextData;
    protected NDManager manager;
    protected Usage usage;

    protected MRL mrl;
    protected boolean prepared;

    /**
     * Creates a new instance of {@link SpeechRecognitionDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public SpeechRecognitionDataset(AudioBuilder<?> builder) {
        super(builder);
        sourceAudioData =
                new AudioData(getDefaultConfiguration().update(builder.sourceConfiguration));
        targetTextData =
                new TextData(
                        TextData.getDefaultConfiguration().update(builder.targetConfiguration));
        manager = builder.manager;
        usage = builder.usage;
    }

    protected void targetPreprocess(List<String> newTextData, boolean source)
            throws EmbeddingException {
        TextData textData = targetTextData;
        textData.preprocess(
                manager, newTextData.subList(0, (int) Math.min(limit, newTextData.size())));
    }

    //    public NDArray getProcessedData(long index, boolean source){
    //        TextData textData = targetTextData;
    //        AudioData audioData = sourceAudioData;
    //        return source ? textData.getRawText(index) : audioData.getPreprocessedAudio(index,
    // manager);
    //    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {}

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        return null;
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    public static AudioData.Configuration getDefaultConfiguration() {
        return new AudioData.Configuration();
    }

    /** Abstract AudioBuilder that helps build a {@link SpeechRecognitionDataset}. */
    public abstract static class AudioBuilder<T extends AudioBuilder<T>> extends BaseBuilder<T> {

        protected AudioData.Configuration sourceConfiguration;
        protected TextData.Configuration targetConfiguration;
        protected NDManager manager;

        protected Repository repository;
        protected String groupId;
        // protected String artifactId;
        protected Usage usage;

        /** Constructs a new builder. */
        AudioBuilder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            usage = Usage.TRAIN;
            sourceConfiguration = new AudioData.Configuration();
            targetConfiguration = new TextData.Configuration();
            manager = Engine.getInstance().newBaseManager();
        }

        /**
         * Sets the {@link AudioData.Configuration} to use for the source text data.
         *
         * @param sourceConfiguration the {@link AudioData.Configuration}
         * @return this builder
         */
        public T setSourceConfiguration(AudioData.Configuration sourceConfiguration) {
            this.sourceConfiguration = sourceConfiguration;
            return self();
        }

        /**
         * Sets the {@link TextData.Configuration} to use for the target text data.
         *
         * @param targetConfiguration the {@link TextData.Configuration}
         * @return this builder
         */
        public T setTargetConfiguration(TextData.Configuration targetConfiguration) {
            this.targetConfiguration = targetConfiguration;
            return self();
        }

        /**
         * Sets the optional manager for the dataset (default follows engine default).
         *
         * @param manager the manager
         * @return this builder
         */
        public T optManager(NDManager manager) {
            this.manager = manager.newSubManager();
            return self();
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public T optUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public T optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId}
         * @return this builder
         */
        public T optGroupId(String groupId) {
            this.groupId = groupId;
            return self();
        }
    }
}
