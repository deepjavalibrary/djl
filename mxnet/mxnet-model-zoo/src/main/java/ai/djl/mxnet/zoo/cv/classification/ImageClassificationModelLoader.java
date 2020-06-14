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
package ai.djl.mxnet.zoo.cv.classification;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.wrapper.FileTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.InputStreamTranslatorFactory;
import ai.djl.modality.cv.translator.wrapper.UrlTranslatorFactory;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;

/** Model loader for Image Classification models. */
public abstract class ImageClassificationModelLoader
        extends BaseModelLoader<Image, Classifications> {

    private static final Application APPLICATION = Application.CV.IMAGE_CLASSIFICATION;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param artifactId the artifact id of the model
     * @param version the version number of the model
     */
    public ImageClassificationModelLoader(
            Repository repository, String artifactId, String version) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, artifactId), version, new MxModelZoo());
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(Image.class, Classifications.class), factory);
        factories.put(
                new Pair<>(Path.class, Classifications.class),
                new FileTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(URL.class, Classifications.class), new UrlTranslatorFactory<>(factory));
        factories.put(
                new Pair<>(InputStream.class, Classifications.class),
                new InputStreamTranslatorFactory<>(factory));
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    /**
     * Loads the model with the given search filters.
     *
     * @param filters the search filters to match against the loaded model
     * @param device the device the loaded model should use
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    @Override
    public ZooModel<Image, Classifications> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, Classifications> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, Classifications> newInstance(Map<String, Object> arguments) {
            int width = ((Double) arguments.getOrDefault("width", 224d)).intValue();
            int height = ((Double) arguments.getOrDefault("height", 224d)).intValue();
            String flag = (String) arguments.getOrDefault("flag", Image.Flag.COLOR.name());

            return ImageClassificationTranslator.builder()
                    .optFlag(Image.Flag.valueOf(flag))
                    .addTransform(new CenterCrop())
                    .addTransform(new Resize(width, height))
                    .addTransform(new ToTensor())
                    .optApplySoftmax(true)
                    .build();
        }
    }
}
