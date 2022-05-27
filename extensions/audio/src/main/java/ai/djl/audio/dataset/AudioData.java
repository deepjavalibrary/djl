package ai.djl.audio.dataset;

import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacv.*;

public class AudioData {
    private List<Float> floatList;
    private Integer sampleRate;
    private Integer audioChannels;

    public AudioData(Configuration configuration){
        this.audioChannels = configuration.audioChannels;
        this.sampleRate = configuration.sampleRate;
        this.toFloat(configuration.path);
    }

    public Integer getAudioChannels() {
        return audioChannels;
    }

    public List<Float> getFloatList() {
        return floatList;
    }

    public Integer getSampleRate() {
        return sampleRate;
    }

    public void setFloatList(List<Float> floatList) {
        this.floatList = floatList;
    }

    public void setAudioChannels(Integer channels) {
        this.audioChannels = channels;
    }

    private AudioData toFloat(String path){
        List<Float> list = new ArrayList<>();
        float scale = (float) 1.0 / (float) (1 << (8 * 2) - 1);
        System.out.println("test");
        try (FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)) {
            audioGrabber.start();
            audioGrabber.setSampleRate(sampleRate);
            setAudioChannels(audioGrabber.getAudioChannels());
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
        setFloatList(list);
        return this;
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


    /**
     * The configuration for creating a {@link AudioData} value in a {@link
     * * ai.djl.training.dataset.Dataset}.
     */
    public static final class Configuration {

        /**
         * This parameter is used for setting normalized value.
         */
        private Double target_dB;
        /**
         * This parameter is used for setting stride value.
         */
        private Double stride_ms;
        /**
         * This parameter is used for setting window frame size value.
         */
        private Double windows_ms;

        private boolean isFeaturize;

        private Integer sampleRate;

        private Integer audioChannels;

        private String path;


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


        public Configuration setFeaturize(boolean featurize) {
            isFeaturize = featurize;
            return this;
        }


        public Configuration setSampleRate(Integer sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }


        public Configuration setAudioChannels(Integer audioChannels) {
            this.audioChannels = audioChannels;
            return this;
        }

        public Configuration setPath(String path) {
            this.path = path;
            return this;
        }

        public AudioData.Configuration update(AudioData.Configuration other) {
            target_dB = other.target_dB;
            stride_ms = other.stride_ms;
            windows_ms = other.windows_ms;
            isFeaturize = other.isFeaturize;
            sampleRate = other.sampleRate;
            audioChannels= other.audioChannels;
            return this;
        }
    }

}
