package ai.djl.audio.dataset;

import java.util.List;
import org.bytedeco.javacv.FrameGrabber;
import org.testng.annotations.Test;

public class testAudioData {

    @Test
    public void testAudioDataTofloat() throws FrameGrabber.Exception {
        String path = "src/test/resources/61-70968-0000.flac";
        AudioData.Configuration configuration = new AudioData.Configuration();
        configuration.setSampleRate(40000);
        AudioData audioData = new AudioData(configuration);
        List<Float> floatList = audioData.getFloatList();
        for (float f : floatList) {
            System.out.println(f);
        }
        System.out.println(floatList.size());
    }
}
