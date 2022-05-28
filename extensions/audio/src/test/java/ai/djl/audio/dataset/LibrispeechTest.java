package ai.djl.audio.dataset;

import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.List;

import org.testng.annotations.Test;

public class LibrispeechTest {
    @Test
    public static void testLibrispeech() throws IOException, TranslateException {

        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        NDManager manager = NDManager.newBaseManager();
        Librispeech dataset =
                Librispeech.builder()
                        .optRepository(repository)
                        .optUsage(Dataset.Usage.TEST)
                        .setSampling(32, true)
                        .build();
        dataset.prepare();
        List<String> list = dataset.sourceAudioData.getAudioPaths();
        for (String path: list) {
            System.out.println(path);
        }
        System.out.println(dataset.get(manager,0).getData());
        System.out.println(dataset.get(manager,0).getLabels());
    }
}
