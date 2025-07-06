/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.openai;

import java.util.List;

/** The chat completion style output. */
public class ChatOutput {

    private String id;
    private String object;
    private Long created;
    private List<Choice> choices;
    private String model;
    private Usage usage;

    ChatOutput() {}

    ChatOutput(
            String id,
            String object,
            Long created,
            List<Choice> choices,
            String model,
            Usage usage) {
        this.id = id;
        this.object = object;
        this.created = created;
        this.choices = choices;
        this.model = model;
        this.usage = usage;
    }

    /**
     * Returns the id.
     *
     * @return the id
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the object.
     *
     * @return the object
     */
    public String getObject() {
        return object;
    }

    /**
     * Returns the created time.
     *
     * @return the created time
     */
    public Long getCreated() {
        return created;
    }

    /**
     * Returns the choices.
     *
     * @return the choices
     */
    public List<Choice> getChoices() {
        return choices;
    }

    /**
     * Returns the model name.
     *
     * @return the model name
     */
    public String getModel() {
        return model;
    }

    /**
     * Returns the usage.
     *
     * @return the usage
     */
    public Usage getUsage() {
        return usage;
    }

    /**
     * Returns the aggregated text output.
     *
     * @return the aggregated text output
     */
    @SuppressWarnings("unchecked")
    public String getTextOutput() {
        if (choices == null || choices.isEmpty()) {
            return "";
        }
        Message message = choices.get(0).getMessage();
        if (message == null) {
            message = choices.get(0).getDelta();
        }
        if (message == null) {
            return "";
        }
        Object content = message.getContent();
        if (content instanceof String) {
            return (String) content;
        } else if (content instanceof List) {
            List<Content> contents = (List<Content>) content;
            StringBuilder sb = new StringBuilder();
            for (Content part : contents) {
                if ("text".equals(part.getType())) {
                    sb.append(part.getText());
                }
            }
            return sb.toString();
        }
        return "";
    }
}
