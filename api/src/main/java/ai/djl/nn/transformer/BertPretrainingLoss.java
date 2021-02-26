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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDList;
import ai.djl.training.loss.AbstractCompositeLoss;
import ai.djl.util.Pair;
import java.util.Arrays;

/** Loss that combines the next sentence and masked language losses of bert pretraining. */
public class BertPretrainingLoss extends AbstractCompositeLoss {

    private BertNextSentenceLoss bertNextSentenceLoss = new BertNextSentenceLoss(0, 0);
    private BertMaskedLanguageModelLoss bertMaskedLanguageModelLoss =
            new BertMaskedLanguageModelLoss(1, 2, 1);

    /** Creates a loss combining the next sentence and masked language loss for bert pretraining. */
    public BertPretrainingLoss() {
        super(BertPretrainingLoss.class.getSimpleName());
        this.components = Arrays.asList(bertNextSentenceLoss, bertMaskedLanguageModelLoss);
    }

    @Override
    protected Pair<NDList, NDList> inputForComponent(
            int componentIndex, NDList labels, NDList predictions) {
        return new Pair<>(labels, predictions);
    }

    /**
     * gets BertNextSentenceLoss.
     *
     * @return BertNextSentenceLoss
     */
    public BertNextSentenceLoss getBertNextSentenceLoss() {
        return bertNextSentenceLoss;
    }

    /**
     * gets BertMaskedLanguageModelLoss.
     *
     * @return BertMaskedLanguageModelLoss
     */
    public BertMaskedLanguageModelLoss getBertMaskedLanguageModelLoss() {
        return bertMaskedLanguageModelLoss;
    }
}
