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
package ai.djl.test.mock;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class EchoTranslator<T> implements Translator<T, T> {

    private NDList preprocessResult;
    private T output;
    private TranslateException inputException;
    private TranslateException outputException;

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, T input) throws TranslateException {
        if (inputException != null) {
            throw inputException;
        }
        output = input;
        return preprocessResult;
    }

    /** {@inheritDoc} */
    @Override
    public T processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        if (outputException != null) {
            throw outputException;
        }
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    public void setPreprocessResult(NDList preprocessResult) {
        this.preprocessResult = preprocessResult;
    }

    public void setOutput(T output) {
        this.output = output;
    }

    public void setInputException(TranslateException inputException) {
        this.inputException = inputException;
    }

    public void setOutputException(TranslateException outputException) {
        this.outputException = outputException;
    }
}
