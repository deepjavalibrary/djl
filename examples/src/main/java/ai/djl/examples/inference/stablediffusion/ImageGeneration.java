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
package ai.djl.examples.inference.stablediffusion;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class ImageGeneration {

    private ImageGeneration() {}

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        // Use Device.gpu() if you need to run on GPU
        StableDiffusionModel model = new StableDiffusionModel(Device.cpu());
        Image result =
                model.generateImageFromText(
                        "Photograph of an astronaut riding a horse in desert", 50);
        saveImage(result, "generated", "build/output");
    }

    public static void saveImage(Image image, String name, String path) throws IOException {
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }
}
