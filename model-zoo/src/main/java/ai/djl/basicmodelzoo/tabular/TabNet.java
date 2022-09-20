package ai.djl.basicmodelzoo.tabular;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.GhostBatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class TabNet {


    public Block FeatureBlock(float batchNormMomentum,long featureDim,Block fc){
        SequentialBlock featureBlock = new SequentialBlock();
        if(fc.isInitialized()){
            featureBlock.add(fc);
        }else{
            featureBlock.add(Linear.builder().setUnits(featureDim*2).build());
        }

        featureBlock.add(GhostBatchNorm.builder().optMomentum(batchNormMomentum).build());
        featureBlock.add(Activation.swishBlock(1f));
        return featureBlock;
    }

    public static class Builder{
        float batchNormMomentum = 0.9f;

    }
}
