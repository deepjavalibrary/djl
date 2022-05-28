package ai.djl.audio.dataset;

import ai.djl.audio.featurizer.AudioNormalizer;
import ai.djl.audio.featurizer.AudioProcessor;
import ai.djl.audio.featurizer.SpecgramFeaturizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.bytedeco.javacv.*;

public class AudioData {
    private int sampleRate;
    private int audioChannels;

    private List<AudioProcessor> processorList;

    public AudioData(Configuration configuration) {
        this.processorList = configuration.processorList;
    }

    /**
     * Returns a good default {@link AudioData.Configuration} to use for the constructor with
     * defaults.
     *
     * @return a good default {@link AudioData.Configuration} to use for the constructor with
     *     defaults
     */
    public static AudioData.Configuration getDefaultConfiguration() {
        List<AudioProcessor> defaultProcessors =
                Arrays.asList(new AudioNormalizer(), new SpecgramFeaturizer());
        return new AudioData.Configuration().setProcessorList(defaultProcessors);
    }

    private float[] toFloat(String path) {
        List<Float> list = new ArrayList<>();
        float scale = (float) 1.0 / (float) (1 << (8 * 2) - 1);
        System.out.println("test");
        try (FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)) {
            audioGrabber.start();
            audioChannels = audioGrabber.getAudioChannels();
            sampleRate = audioGrabber.getSampleRate();
            Frame frame;
            while ((frame = audioGrabber.grabFrame()) != null) {
                Buffer[] buffers = frame.samples;
                //                ShortBuffer[] copiedBuffer = new ShortBuffer[buffers.length];
                //                for (int i = 0; i < buffers.length; i++) {
                //                    deepCopy(buffers[i], copiedBuffer[i]);
                //                }
                ShortBuffer sb = (ShortBuffer) buffers[0];
                for (int i = 0; i < sb.limit(); i++) {
                    list.add(sb.get() * scale);
                }
            }
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        float[] floatArray = new float[list.size()];
        int i = 0;
        for (Float f : list) {
            floatArray[i++] = (f != null ? f : Float.NaN);
        }
        return floatArray;
    }

    public NDArray getPreprocessedData(NDManager manager, String path) {
        float[] floatArray = toFloat(path);
        NDArray samples = manager.create(floatArray);
        for (AudioProcessor processor : processorList) {
            samples = processor.ExtractFeatures(manager, samples);
        }
        return samples;
    }

    private static ShortBuffer deepCopy(ShortBuffer source, ShortBuffer target) {
        int sourceP = source.position();
        int sourceL = source.limit();
        if (null == target) {
            target = ShortBuffer.allocate(source.remaining());
        }
        target.put(source);
        target.flip();
        source.position(sourceP);
        source.limit(sourceL);
        return target;
    }

    public int getAudioChannels() {
        return audioChannels;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    /**
     * The configuration for creating a {@link AudioData} value in a {@link *
     * ai.djl.training.dataset.Dataset}.
     */
    public static final class Configuration {

        /** This parameter is used for setting normalized value. */
        private Double target_dB;
        /** This parameter is used for setting stride value. */
        private Double stride_ms;
        /** This parameter is used for setting window frame size value. */
        private Double windows_ms;
        private List<AudioProcessor> processorList;

        public Configuration setStride_ms(Double stride_ms) {
            this.stride_ms = stride_ms;
            return this;
        }

        public Configuration setTarget_dB(Double target_dB) {
            this.target_dB = target_dB;
            return this;
        }

        public Configuration setWindows_ms(Double windows_ms) {
            this.windows_ms = windows_ms;
            return this;
        }

        public Configuration setProcessorList(List<AudioProcessor> processorList) {
            this.processorList = processorList;
            return this;
        }

        public AudioData.Configuration update(AudioData.Configuration other) {
            target_dB = other.target_dB;
            stride_ms = other.stride_ms;
            windows_ms = other.windows_ms;
            processorList = other.processorList;
            return this;
        }
    }
}
