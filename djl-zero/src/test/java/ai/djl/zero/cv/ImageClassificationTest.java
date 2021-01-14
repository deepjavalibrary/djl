/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.zero.cv;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.zero.Performance;
import ai.djl.zero.cv.ImageClassification.Classes;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class ImageClassificationTest {

    @Test
    public void testImageClassificationPretrained()
            throws IOException, ModelNotFoundException, MalformedModelException {
        Class<?>[] inputClasses = {Image.class, Path.class, URL.class, InputStream.class};
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        for (Class<?> inputClass : inputClasses) {
            for (Classes classes : Classes.values()) {
                for (Performance performance : Performance.values()) {
                    ZooModel<?, ?> model =
                            ImageClassification.pretrained(inputClass, classes, performance);
                    model.close();
                }
            }
        }
    }
}
