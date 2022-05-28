package ai.djl.audio.dataset;

import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.annotations.Test;

public class LibrispeechTest {
    @Test
    public static void testLibrispeech() throws IOException, TranslateException {

        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");

        Librispeech dataset =
                Librispeech.builder()
                        .optRepository(repository)
                        .optUsage(Dataset.Usage.TEST)
                        .setSampling(32, true)
                        .build();
        dataset.prepare();
    }
    //        for (Dataset.Usage usage :
    //                new Dataset.Usage[] {
    //                        Dataset.Usage.TRAIN, Dataset.Usage.TEST
    //                }) {
    //            try (NDManager manager = NDManager.newBaseManager()) {
    //                Librispeech dataset =
    //                        Librispeech.builder()
    //                                .setSourceConfiguration(
    //                                        new AudioData.Configuration())
    //                                .setTargetConfiguration(
    //                                        new TextData.Configuration()
    //                                                .setTextEmbedding(
    //                                                        TestUtils.getTextEmbedding(
    //                                                                manager, EMBEDDING_SIZE))
    //                                                .setEmbeddingSize(EMBEDDING_SIZE))
    //                                .setSampling(32, true)
    //                                .optLimit(100)
    //                                .optUsage(usage)
    //                                .build();
    //                dataset.prepare();
    //                Record record = dataset.get(manager, 0);
    //                Assert.assertEquals(record.getData().get(0).getShape().get(1), 15);
    //                Assert.assertNull(record.getLabels());
    //            }
    //        }
    //    }
}
