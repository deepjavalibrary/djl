package ai.djl.basicdataset.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.bytedeco.javacv.*;

import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.List;

public class AudioData {
    private float[] samples = null;
    private Integer sampleRate = -1;
    private Integer audioChannels = -1;

    public Integer getAudioChannels() {
        return audioChannels;
    }

    public float[] getSamples() {
        return samples;
    }

    public Integer getSampleRate() {
        return sampleRate;
    }
    public void setSamples(float[] floatArray){
        this.samples = floatArray;
    }
    public void setSampleRate(Integer rate){
        this.sampleRate = rate;
    }
    public void setAudioChannels(Integer channels){
        this.audioChannels = channels;
    }

    public AudioData toFloat(String path) throws FrameGrabber.Exception {
        List<Float> floatList = new ArrayList<>();
        AudioData audioData= new AudioData();
        audioData.setAudioChannels(-1);
        audioData.setSampleRate(-1);
        float scale = (float) 1.0/ (float) (1 << (8 * 2) - 1);
        try(FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)){
            audioGrabber.start();
            Frame frame;
            while((frame = audioGrabber.grabFrame()) != null){
                Buffer[] buffers = frame.samples;
                ShortBuffer sb = (ShortBuffer) buffers[0];
                floatList.add (sb.get() * scale);
            }
        }
        catch (FrameGrabber.Exception e){
            e.printStackTrace();
        }
        float [] floatArray = new float[floatList.size()];
        for (int i = 0; i < floatArray.length ; i++) {
            floatArray[i] = floatList.get(i);
        }
        audioData.setSamples(floatArray);
        return audioData;
    }


    /**
     * The configuration for creating a {@link AudioData} value in a {@link
     *      * ai.djl.training.dataset.Dataset}.
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

    }

}
