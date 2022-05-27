package ai.djl.audio.dataset;

import org.bytedeco.javacv.FrameGrabber;
import org.testng.annotations.Test;

public class testAudioData {
    @Test
    public void testAudioDatatofloat() throws FrameGrabber.Exception {
        AudioData audioData = new AudioData();
        audioData = audioData.toFloat("src/test/resources/61-70968-0000.flac");
        float[] floats = audioData.getSamples();
        for (float aFloat : floats) {
            System.out.println(aFloat);
        }
    }
}
