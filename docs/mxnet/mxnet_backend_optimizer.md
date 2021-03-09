# Custom backend optimizer support on Apache MXNet

Apache MXNet currently implemented a method that allowing third-party
backend optimizer to accelerate the inference result. DJL currently
also exposed this functionality through the `MxOptimizeFor` option
of the Criteria.

```
.optOption("MxOptimizeFor", "optimizer_name")
```

After a name is passed, DJL will try to find the party library from
 the environment variable called `MXNET_EXTRA_LIBRARY_PATH`. Users are required to
set this environment variable to locate the library.  After that, you should see the messages from the inference to see if the library is enabled.

Here is a list of supporting backend optimizers:

## AWS [Elastic Inference Accelerator](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/what-is-ei.html) (EIA)

Currently, you can use EIA library for DJL on all EI enabled instance.

You can follow the instruction to start your EI application with DJL:

```
> https://docs.aws.amazon.com/elastic-inference/latest/developerguide/ei-mxnet.html
```

Currently, the EI logging is disabled. For debugging purpose, you can enable that through
setting the `MXNET_EXTRA_LIBRARY_VERBOSE` environment variable:

```
export MXNET_EXTRA_LIBRARY_VERBOSE=true
```
