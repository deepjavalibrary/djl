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
package ai.djl.mxnet.zoo.nlp.qa;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

/** The translator for {@link BertQAModelLoader}. */
public class BertQATranslator implements Translator<QAInput, String> {

    private List<String> tokens;

    BertQATranslator() {}

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) throws IOException {
        BertDataParser parser = ctx.getModel().getArtifact("vocab.json", BertDataParser::parse);
        // pre-processing - tokenize sentence
        List<String> tokenQ = BertDataParser.tokenizer(input.getQuestion().toLowerCase());
        List<String> tokenA = BertDataParser.tokenizer(input.getParagraph().toLowerCase());
        int validLength = tokenQ.size() + tokenA.size();
        List<Float> tokenTypes = BertDataParser.getTokenTypes(tokenQ, tokenA, input.getSeqLength());
        tokens = BertDataParser.formTokens(tokenQ, tokenA, input.getSeqLength());
        List<Integer> indexes = parser.token2idx(tokens);
        float[] types = Utils.toFloatArray(tokenTypes);
        float[] indexesFloat = Utils.toFloatArray(indexes);

        int seqLength = input.getSeqLength();
        NDManager manager = ctx.getNDManager();
        NDArray data0 = manager.create(indexesFloat, new Shape(1, seqLength));
        data0.setName("data0");
        NDArray data1 = manager.create(types, new Shape(1, seqLength));
        data1.setName("data1");
        NDArray data2 = manager.create(new float[] {validLength});
        data2.setName("data2");

        return new NDList(data0, data1, data2);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray array = list.singletonOrThrow();
        NDList output = array.split(2, 2);
        // Get the formatted logits result
        NDArray startLogits = output.get(0).reshape(new Shape(1, -1));
        NDArray endLogits = output.get(1).reshape(new Shape(1, -1));
        // Get Probability distribution
        NDArray startProb = startLogits.softmax(-1);
        NDArray endProb = endLogits.softmax(-1);
        int startIdx = (int) startProb.argMax(1).getFloat();
        int endIdx = (int) endProb.argMax(1).getFloat();
        return tokens.subList(startIdx, endIdx + 1).toString();
    }
}
