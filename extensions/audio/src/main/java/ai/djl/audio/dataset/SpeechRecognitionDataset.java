package ai.djl.audio.dataset;

import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.IOException;

public abstract class SpeechRecognitionDataset extends RandomAccessDataset {

    protected AudioData sourceAudioData;
    protected TextData recognition;
    protected NDManager manager;
    protected Usage usage;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public SpeechRecognitionDataset(BaseBuilder<?> builder) {
        super(builder);
    }

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

    public abstract static class Builder<T extends Builder<T>> extends BaseBuilder<T> {}
}
