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
package ai.djl.mxnet.zoo.nlp.bertqa;

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

public class BertQATranslator implements Translator<QAInput, String> {

    private List<String> tokens;

    BertQATranslator() {}

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) throws IOException {
        BertDataParser parser = ctx.getModel().getArtifact("vocab.json", BertDataParser::parse);
        // pre-processing - tokenize sentence
        List<String> tokenQ = BertDataParser.tokenizer(input.getQuestion().toLowerCase());
        List<String> tokenA = BertDataParser.tokenizer(input.getAnswer().toLowerCase());
        int validLength = tokenQ.size() + tokenA.size();
        List<Float> tokenTypes = BertDataParser.getTokenTypes(tokenQ, tokenA, input.getSeqLength());
        tokens = BertDataParser.formTokens(tokenQ, tokenA, input.getSeqLength());
        List<Integer> indexes = parser.token2idx(tokens);
        float[] types = Utils.toFloatArray(tokenTypes);
        float[] indexesFloat = Utils.toFloatArray(indexes);

        int seqLength = input.getSeqLength();
        NDManager manager = ctx.getNDManager();
        NDArray data0 = manager.create(indexesFloat, new Shape(1, seqLength));
        NDArray data1 = manager.create(types, new Shape(1, seqLength));
        NDArray data2 = manager.create(new float[] {validLength});

        return new NDList(3).add("data0", data0).add("data1", data1).add("data2", data2);
    }

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
        int startIdx = (int) startProb.argmax(1).getFloat();
        int endIdx = (int) endProb.argmax(1).getFloat();
        return tokens.subList(startIdx, endIdx + 1).toString();
    }
}
