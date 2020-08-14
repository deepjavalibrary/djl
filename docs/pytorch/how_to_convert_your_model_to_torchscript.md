## How to convert your PyTorch model to TorchScript

There are two ways to convert your model to TorchScript: tracing and scripting.
We will only demonstrate the first one, tracing, but you can find information about scripting from the PyTorch documentation.
When tracing, we use an example input to record the actions taken and capture the the model architecture.
This works best when your model doesn't have control flow. 
If you do have control flow, you will need to use the scripting approach.
In DJL, we use tracing to create TorchScript for our ModelZoo models.

Here is an example of tracing in actions:

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")
```

For more information on exporting TorchScript, see the [Loading a torch script model in C++ tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html).
