/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicmodelzoo.cv.classification;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.ImageClassificationTranslator;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Model loader for MLP models. */
public class MlpModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = BasicModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "mlp";
    private static final String VERSION = "0.0.2";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public MlpModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        Map<Type, TranslatorFactory<?, ?>> map = new ConcurrentHashMap<>();
        map.put(Classifications.class, new FactoryImpl());
        factories.put(BufferedImage.class, map);
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    /** {@inheritDoc} */
    @Override
    public Model loadModel(Artifact artifact, Device device, Map<String, Object> override)
            throws IOException, MalformedModelException {
        Map<String, Object> arguments = artifact.getArguments(override);
        int width = ((Double) arguments.getOrDefault("width", 28d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 28d)).intValue();
        int input = width * height;
        int output = ((Double) arguments.get("output")).intValue();
        @SuppressWarnings("unchecked")
        int[] hidden =
                ((List<Double>) arguments.get("hidden"))
                        .stream()
                        .mapToInt(Double::intValue)
                        .toArray();

        Path dir = repository.getCacheDirectory();
        String relativePath = artifact.getResourceUri().getPath();
        Path modelPath = dir.resolve(relativePath);

        Model model = Model.newInstance(device);
        model.setBlock(new Mlp(input, output, hidden));
        model.load(modelPath, artifact.getName());
        return model;
    }

    private static final class FactoryImpl
            implements TranslatorFactory<BufferedImage, Classifications> {

        @Override
        public Translator<BufferedImage, Classifications> newInstance(
                Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 28d)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 28d)).intValue();
            String flag = (String) arguments.getOrDefault("flag", NDImageUtils.Flag.COLOR.name());

            Pipeline pipeline = new Pipeline();
            pipeline.add(new CenterCrop()).add(new Resize(width, height)).add(new ToTensor());
            return ImageClassificationTranslator.builder()
                    .optFlag(NDImageUtils.Flag.valueOf(flag))
                    .setPipeline(pipeline)
                    .setSynsetArtifactName("synset.txt")
                    .build();
        }
    }
}
