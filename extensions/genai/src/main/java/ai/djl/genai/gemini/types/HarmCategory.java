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
package ai.djl.genai.gemini.types;

/** An enum represent Gemini schema. */
public enum HarmCategory {
    HARM_CATEGORY_UNSPECIFIED,
    HARM_CATEGORY_HATE_SPEECH,
    HARM_CATEGORY_DANGEROUS_CONTENT,
    HARM_CATEGORY_HARASSMENT,
    HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HARM_CATEGORY_CIVIC_INTEGRITY,
    HARM_CATEGORY_IMAGE_HATE,
    HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT,
    HARM_CATEGORY_IMAGE_HARASSMENT,
    HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT
}
