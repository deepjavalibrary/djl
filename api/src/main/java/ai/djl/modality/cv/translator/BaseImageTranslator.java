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
package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.List;

/**
 * Built-in {@code Translator} that provides default image pre-processing.
 *
 * @param <T> the output object type
 */
public abstract class BaseImageTranslator<T> implements Translator<Image, T> {

    private Image.Flag flag;
    private Pipeline pipeline;
    private Batchifier batchifier;

    /**
     * Constructs an ImageTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    public BaseImageTranslator(BaseBuilder<?> builder) {
        flag = builder.flag;
        pipeline = builder.pipeline;
        batchifier = builder.batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Pipeline getPipeline() {
        return pipeline;
    }

    /**
     * Processes the {@link Image} input and converts it to NDList.
     *
     * @param ctx the toolkit that helps create the input NDArray
     * @param input the {@link Image} input
     * @return a {@link NDList}
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager(), flag);
        return pipeline.transform(new NDList(array));
    }

    /**
     * A builder to extend for all classes extending the {@link BaseImageTranslator}.
     *
     * @param <T> the concrete builder type
     */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Image.Flag flag = Image.Flag.COLOR;
        protected Pipeline pipeline;
        protected Batchifier batchifier = Batchifier.STACK;

        /**
         * Sets the optional {@link ai.djl.modality.cv.Image.Flag} (default is {@link
         * Image.Flag#COLOR}).
         *
         * @param flag the color mode for the images
         * @return this builder
         */
        public T optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
        }

        /**
         * Sets the {@link Pipeline} to use for pre-processing the image.
         *
         * @param pipeline the pre-processing pipeline
         * @return this builder
         */
        public T setPipeline(Pipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        /**
         * Adds the {@link Transform} to the {@link Pipeline} use for pre-processing the image.
         *
         * @param transform the {@link Transform} to be added
         * @return this builder
         */
        public T addTransform(Transform transform) {
            if (pipeline == null) {
                pipeline = new Pipeline();
            }
            pipeline.add(transform);
            return self();
        }

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier the {@link Batchifier} to be set
         * @return this builder
         */
        public T optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return self();
        }

        protected abstract T self();

        protected void validate() {
            if (pipeline == null) {
                throw new IllegalArgumentException("pipeline is required.");
            }
        }
    }

    /** A Builder to construct a {@code ImageClassificationTranslator}. */
    @SuppressWarnings("rawtypes")
    public abstract static class ClassificationBuilder<T extends BaseBuilder>
            extends BaseBuilder<T> {

        protected SynsetLoader synsetLoader;

        /**
         * Sets the name of the synset file listing the potential classes for an image.
         *
         * @param synsetArtifactName a file listing the potential classes for an image
         * @return the builder
         */
        public T optSynsetArtifactName(String synsetArtifactName) {
            synsetLoader = new SynsetLoader(synsetArtifactName);
            return self();
        }

        /**
         * Sets the URL of the synset file.
         *
         * @param synsetUrl the URL of the synset file
         * @return the builder
         */
        public T optSynsetUrl(URL synsetUrl) {
            this.synsetLoader = new SynsetLoader(synsetUrl);
            return self();
        }

        /**
         * Sets the potential classes for an image.
         *
         * @param synset the potential classes for an image
         * @return the builder
         */
        public T optSynset(List<String> synset) {
            synsetLoader = new SynsetLoader(synset);
            return self();
        }

        /** {@inheritDoc} */
        @Override
        protected void validate() {
            super.validate();
            if (synsetLoader == null) {
                synsetLoader = new SynsetLoader("synset.txt");
            }
        }
    }

    protected static final class SynsetLoader {

        private String synsetFileName;
        private URL synsetUrl;
        private List<String> synset;

        public SynsetLoader(List<String> synset) {
            this.synset = synset;
        }

        public SynsetLoader(URL synsetUrl) {
            this.synsetUrl = synsetUrl;
        }

        public SynsetLoader(String synsetFileName) {
            this.synsetFileName = synsetFileName;
        }

        public List<String> load(Model model) throws IOException {
            if (synset != null) {
                return synset;
            } else if (synsetUrl != null) {
                try (InputStream is = synsetUrl.openStream()) {
                    return Utils.readLines(is);
                }
            }
            return model.getArtifact(synsetFileName, Utils::readLines);
        }
    }
}
