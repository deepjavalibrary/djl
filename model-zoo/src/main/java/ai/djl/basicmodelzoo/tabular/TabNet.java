package ai.djl.basicmodelzoo.tabular;

import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.GhostBatchNorm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TabNet {

    /**
     * Creates a FL->BN->GLU block used in tabNet
     * @param fc the fully connected layer
     * @param featureDim
     * @param applyGLU
     * @param batchNormMomentum
     * @return a FL->BN->GLU block
     */
    public static Block featureBlock(Block fc,int featureDim,boolean applyGLU,float batchNormMomentum){
        SequentialBlock featureBlock = new SequentialBlock();
        int units = applyGLU?2*featureDim:featureDim;
        if(fc==null){
            featureBlock.add(Linear.builder().setUnits(units).build());
        }else{
            featureBlock.add(fc);
        }

        featureBlock.add(GhostBatchNorm.builder()
                .optMomentum(batchNormMomentum)
                .build());
        if(applyGLU){
            featureBlock.add(Activation.tabNetGLUBlock(units));
        }
        return featureBlock;
    }

    public static Block featureTransformer(List<Block> fcs,int featureDim,int nTotal,float batchNormMomentum){
        List<Block> blocks = new ArrayList<>();
        for(int i = 0;i<nTotal;i++){
            if(!fcs.isEmpty()&&i<fcs.size()){
                blocks.add(featureBlock(fcs.get(i),featureDim,true,batchNormMomentum));
            }else{
                blocks.add(featureBlock(null,featureDim,true,batchNormMomentum));
            }
        }
        SequentialBlock featureTransformer = new SequentialBlock();
        featureTransformer.add(blocks.get(0));
        for(int i = 1;i<nTotal;i++){
            featureTransformer.add(
                    new ParallelBlock(
                            list->new NDList(
                                    NDArrays.add(
                                            list.get(0).singletonOrThrow(),
                                            list.get(1).singletonOrThrow().mul(Math.sqrt(0.5))
                                    )
                            ), Arrays.asList(blocks.get(i),Blocks.identityBlock())
                    )
            );
        }
        return featureTransformer;
    }

    public static Block attentiveTransformer(int featureDim,float batchNormMomentum){
        return new SequentialBlock().add(
                featureBlock(null,featureDim,false,batchNormMomentum)
        );
    }

    public static class Builder{
        int outDim = 0;
    }
}
