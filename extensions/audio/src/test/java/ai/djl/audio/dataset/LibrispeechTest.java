/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.audio.dataset;

import ai.djl.ndarray.NDManager;
import ai.djl.repository.Repository;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.annotations.Test;

public class LibrispeechTest {
    @Test(enabled = false)
    public void testLibrispeech() throws IOException, TranslateException {

        Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo");
        NDManager manager = NDManager.newBaseManager();
        Librispeech dataset =
                Librispeech.builder()
                        .optRepository(repository)
                        .optUsage(Dataset.Usage.TEST)
                        .setSampling(32, true)
                        .build();
        dataset.prepare();
        System.out.println(dataset.get(manager, 0).getData());
        System.out.println(dataset.sourceAudioData.getAudioPaths().get(0));
        for (long i : dataset.get(manager, 0).getLabels().get(0).toLongArray()) {
            System.out.print(i);
            System.out.print(" ");
        }
    }
}
