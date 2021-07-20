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
package ai.djl.fasttext.zoo.nlp.textclassification;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.fasttext.FtModel;
import ai.djl.fasttext.zoo.FtModelZoo;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;

/** Model loader for fastText cooking stackexchange models. */
public class TextClassificationModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.NLP.TEXT_CLASSIFICATION;
    private static final String GROUP_ID = FtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "cooking_stackexchange";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public TextClassificationModelLoader(Repository repository) {
        super(repository.model(APPLICATION, GROUP_ID, ARTIFACT_ID, VERSION), new FtModelZoo());
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> ZooModel<I, O> loadModel(Criteria<I, O> criteria)
            throws ModelNotFoundException, IOException, MalformedModelException {
        Artifact artifact = mrl.match(criteria.getFilters());
        if (artifact == null) {
            throw new ModelNotFoundException("No matching filter found");
        }

        Progress progress = criteria.getProgress();
        mrl.prepare(artifact, progress);
        if (progress != null) {
            progress.reset("Loading", 2);
            progress.update(1);
        }
        String modelName = criteria.getModelName();
        if (modelName == null) {
            modelName = artifact.getName();
        }
        Model model = new FtModel(modelName);
        Path modelPath = mrl.getRepository().getResourceDirectory(artifact);
        model.load(modelPath);
        return new ZooModel<>(model, null);
    }
}
