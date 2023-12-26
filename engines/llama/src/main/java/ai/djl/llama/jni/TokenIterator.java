/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.jni;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;

/** A iterator class holds generated tokens. */
public class TokenIterator implements Iterator<Token> {

    private static final Logger logger = LoggerFactory.getLogger(TokenIterator.class);

    private static AtomicBoolean active = new AtomicBoolean();

    private long handle;
    private long count;
    private long pos;
    private boolean hasNext;

    /**
     * Constructs a new {@code TokenIterator} instance.
     *
     * @param handle the llama.cpp handle
     */
    public TokenIterator(long handle) {
        this.handle = handle;
        hasNext = true;
        if (!active.compareAndSet(false, true)) {
            active.set(true);
            logger.warn("Previous inference has been reset");
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasNext() {
        return hasNext;
    }

    /** {@inheritDoc} */
    @Override
    public Token next() {
        if (!hasNext) {
            throw new NoSuchElementException();
        }
        Token token = LlamaLibrary.getNext(handle, count, pos);
        count = token.count;
        pos = token.pos;
        hasNext = token.hasNext;
        if (!hasNext) {
            active.set(false);
        }
        return token;
    }
}
