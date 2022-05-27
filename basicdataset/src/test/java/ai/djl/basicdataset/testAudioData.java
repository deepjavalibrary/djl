package ai.djl.basicdataset;
import ai.djl.basicdataset.utils.AudioData;
import org.bytedeco.javacv.FrameGrabber;
import org.testng.annotations.Test;

public class testAudioData {
    String path = "E:\\OneDrive\\Third_year_second_semester\\Software_Enginering\\final_project\\djl\\basicdataset\\src\\test\\resources\\mlrepo\\dataset\\audio\\61-70968-0000.flac";
    AudioData audioData = new AudioData();
    @Test
    public void testAudioDatatofloat() throws FrameGrabber.Exception {
        audioData = audioData.toFloat(path);
        float[] floats = audioData.getSamples();
        for (float aFloat : floats) {
            System.out.println(aFloat);
        }
    }
}
