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
package ai.djl.translate;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.util.JsonUtils;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Constructor;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link TranslatorFactory} that creates an generic {@link Translator}. */
public class ServingTranslatorFactory implements TranslatorFactory<Input, Output> {

    private static final Logger logger = LoggerFactory.getLogger(ServingTranslatorFactory.class);

    /** {@inheritDoc} */
    @Override
    public Translator<Input, Output> newInstance(Model model, Map<String, Object> arguments)
            throws TranslateException {
        Path modelDir = model.getModelPath();
        Path libPath = modelDir.resolve("libs");
        if (!Files.isDirectory(libPath)) {
            libPath = modelDir.resolve("lib");
            if (!Files.isDirectory(libPath)) {
                return loadDefaultTranslator(model, arguments);
            }
        }
        String className = null;
        Path manifestFile = libPath.resolve("serving.properties");
        if (Files.isRegularFile(manifestFile)) {
            Properties prop = new Properties();
            try (Reader reader = Files.newBufferedReader(manifestFile)) {
                prop.load(reader);
            } catch (IOException e) {
                throw new TranslateException("Failed to load serving.properties file", e);
            }
            className = prop.getProperty("translator");
        }
        ServingTranslator translator = findTranslator(libPath, className);
        if (translator != null) {
            translator.setArguments(arguments);
            return translator;
        }
        return loadDefaultTranslator(model, arguments);
    }

    private ServingTranslator findTranslator(Path path, String className) {
        try {
            List<Path> jarFiles =
                    Files.list(path)
                            .filter(p -> p.toString().endsWith(".jar"))
                            .collect(Collectors.toList());
            List<URL> urls = new ArrayList<>(jarFiles.size() + 1);
            urls.add(path.toUri().toURL());
            for (Path p : jarFiles) {
                urls.add(p.toUri().toURL());
            }

            ClassLoader parentCl = Thread.currentThread().getContextClassLoader();
            ClassLoader cl = new URLClassLoader(urls.toArray(new URL[0]), parentCl);
            if (className != null && !className.isEmpty()) {
                return initTranslator(cl, className);
            }

            ServingTranslator translator = scanDirectory(cl, path);
            if (translator != null) {
                return translator;
            }

            for (Path p : jarFiles) {
                translator = scanJarFile(cl, p);
                if (translator != null) {
                    return translator;
                }
            }
        } catch (IOException e) {
            logger.debug("Failed to find Translator", e);
        }
        return null;
    }

    private ServingTranslator scanDirectory(ClassLoader cl, Path dir) throws IOException {
        Collection<Path> files =
                Files.walk(dir)
                        .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                        .collect(Collectors.toList());
        for (Path file : files) {
            Path p = dir.relativize(file);
            String className = p.toString();
            className = className.substring(0, className.lastIndexOf('.'));
            className = className.replace(File.separatorChar, '.');
            ServingTranslator translator = initTranslator(cl, className);
            if (translator != null) {
                return translator;
            }
        }
        return null;
    }

    private ServingTranslator scanJarFile(ClassLoader cl, Path path) throws IOException {
        try (JarFile jarFile = new JarFile(path.toFile())) {
            Enumeration<JarEntry> en = jarFile.entries();
            while (en.hasMoreElements()) {
                JarEntry entry = en.nextElement();
                String fileName = entry.getName();
                if (fileName.endsWith(".class")) {
                    fileName = fileName.substring(0, fileName.lastIndexOf('.'));
                    fileName = fileName.replace('/', '.');
                    ServingTranslator translator = initTranslator(cl, fileName);
                    if (translator != null) {
                        return translator;
                    }
                }
            }
        }
        return null;
    }

    private ServingTranslator initTranslator(ClassLoader cl, String className) {
        try {
            Class<?> clazz = Class.forName(className, true, cl);
            Class<? extends ServingTranslator> subclass = clazz.asSubclass(ServingTranslator.class);
            Constructor<? extends ServingTranslator> constructor = subclass.getConstructor();
            return constructor.newInstance();
        } catch (Throwable e) {
            logger.trace("Not able to load ModelServerTranslator", e);
        }
        return null;
    }

    private Translator<Input, Output> loadDefaultTranslator(
            Model model, Map<String, Object> arguments) throws TranslateException {
        String appName = model.getProperty("application");
        if (appName != null) {
            Application application = Application.of(appName);
            if (application == Application.CV.IMAGE_CLASSIFICATION) {
                return getImageClassificationTranslator(arguments);
            } else if (application == Application.CV.OBJECT_DETECTION) {
                // TODO: check model name
                return getSsdTranslator(arguments);
            } else {
                // TODO: Add more modalities
                throw new TranslateException("Unsupported application: " + application);
            }
        }
        throw new TranslateException("No ServingTranslator found.");
    }

    private Translator<Input, Output> getImageClassificationTranslator(
            Map<String, Object> arguments) {
        int width = ((Double) arguments.getOrDefault("width", 224d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 224d)).intValue();
        String flag = (String) arguments.getOrDefault("flag", Image.Flag.COLOR.name());

        final Translator<Image, Classifications> translator =
                ImageClassificationTranslator.builder()
                        .optFlag(Image.Flag.valueOf(flag))
                        .addTransform(new CenterCrop())
                        .addTransform(new Resize(width, height))
                        .addTransform(new ToTensor())
                        .build();
        return new ImageServingTranslator(translator);
    }

    private Translator<Input, Output> getSsdTranslator(Map<String, Object> arguments) {
        int width = ((Double) arguments.getOrDefault("width", 512d)).intValue();
        int height = ((Double) arguments.getOrDefault("height", 512d)).intValue();
        double threshold = ((Double) arguments.getOrDefault("threshold", 0.2d));
        String flag = (String) arguments.getOrDefault("flag", Image.Flag.COLOR.name());

        SingleShotDetectionTranslator translator =
                SingleShotDetectionTranslator.builder()
                        .optFlag(Image.Flag.valueOf(flag))
                        .addTransform(new Resize(width, height))
                        .addTransform(new ToTensor())
                        .optThreshold((float) threshold)
                        .optRescaleSize(width, height)
                        .build();
        return new ImageServingTranslator(translator);
    }

    private static final class ImageServingTranslator implements Translator<Input, Output> {

        private Translator<Image, ?> translator;

        public ImageServingTranslator(Translator<Image, ?> translator) {
            this.translator = translator;
        }

        @Override
        public Batchifier getBatchifier() {
            return translator.getBatchifier();
        }

        @Override
        public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
            Input input = (Input) ctx.getAttachment("input");
            Output output = new Output(input.getRequestId(), 200, "OK");
            Object obj = translator.processOutput(ctx, list);
            output.setContent(JsonUtils.GSON_PRETTY.toJson(obj));
            return output;
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
            ctx.setAttachment("input", input);
            byte[] data = input.getContent().valueAt(0);
            Image image =
                    ImageFactory.getInstance().fromInputStream(new ByteArrayInputStream(data));
            return translator.processInput(ctx, image);
        }
    }
}
