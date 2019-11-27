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
package ai.djl.integration.tests.model_zoo;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.mxnet.zoo.MxModelZoo;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class MxModelZooTest {

    @BeforeClass
    public void setUp() {
        // force downloading without cache in .djl.ai folder.
        System.setProperty("DJL_CACHE_DIR", "build/cache");
    }

    @AfterClass
    public void tearDown() {
        System.setProperty("DJL_CACHE_DIR", "");
    }

    @Test
    public void downloadActionRecognitionModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "vgg16");
        criteria.put("dataset", "ucf101");
        Model model = MxModelZoo.ACTION_RECOGNITION.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "inceptionv3");
        criteria.put("dataset", "ucf101");
        model = MxModelZoo.ACTION_RECOGNITION.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadMlpModels() throws IOException, ModelException {
        if (Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("dataset", "mnist");
        Model model = MxModelZoo.MLP.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadResnetModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "18");
        criteria.put("flavor", "v1");
        criteria.put("dataset", "imagenet");
        Model model = MxModelZoo.RESNET.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("layers", "50");
        criteria.put("flavor", "v2");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.RESNET.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("layers", "152");
        criteria.put("flavor", "v1d");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.RESNET.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadResnextModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "101");
        criteria.put("flavor", "64x4d");
        criteria.put("dataset", "imagenet");
        Model model = MxModelZoo.RESNEXT.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadSeResnextModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "101");
        criteria.put("flavor", "32x4d");
        criteria.put("dataset", "imagenet");
        Model model = MxModelZoo.SE_RESNEXT.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("layers", "101");
        criteria.put("flavor", "64x4d");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.SE_RESNEXT.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadSenetModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "154");
        criteria.put("dataset", "imagenet");
        Model model = MxModelZoo.SENET.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadMaskRcnnModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "resnet18");
        criteria.put("flavor", "v1b");
        criteria.put("dataset", "coco");
        Model model = MxModelZoo.MASK_RCNN.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "resnet101");
        criteria.put("flavor", "v1d");
        criteria.put("dataset", "coco");
        model = MxModelZoo.MASK_RCNN.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadSsdModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("size", "512");
        criteria.put("backbone", "resnet50");
        criteria.put("flavor", "v1");
        criteria.put("dataset", "voc");
        Model model = MxModelZoo.SSD.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("size", "512");
        criteria.put("backbone", "vgg16");
        criteria.put("flavor", "atrous");
        criteria.put("dataset", "coco");
        model = MxModelZoo.SSD.loadModel(criteria);
        model.close();
    }

    @Test
    public void downloadSimplePoseModels() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || Boolean.getBoolean("offline")) {
            return;
        }

        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("backbone", "resnet18");
        criteria.put("flavor", "v1b");
        criteria.put("dataset", "imagenet");
        Model model = MxModelZoo.SIMPLE_POSE.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "resnet50");
        criteria.put("flavor", "v1b");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.SIMPLE_POSE.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "resnet101");
        criteria.put("flavor", "v1d");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.SIMPLE_POSE.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "resnet152");
        criteria.put("flavor", "v1b");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.SIMPLE_POSE.loadModel(criteria);
        model.close();

        criteria.clear();
        criteria.put("backbone", "resnet152");
        criteria.put("flavor", "v1d");
        criteria.put("dataset", "imagenet");
        model = MxModelZoo.SIMPLE_POSE.loadModel(criteria);
        model.close();
    }
}
