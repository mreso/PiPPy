# $ torchrun --nproc-per-node 4 pippy_llama.py
import os
import torch
import types
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage

# Grab the model
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)  # bs = 8
tokenizer.pad_token = tokenizer.eos_token

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
llama.to(device).eval()
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# output = llama(inputs["input_ids"])


# Cut model by equal number of layers per rank
layers_per_rank = llama.config.num_hidden_layers // world_size
for i in range(1, world_size):
    annotate_split_points(llama,
        {f"model.layers.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING})

# Create a pipeline representation from the model
llama_pipe = Pipe.from_tracing(llama, world_size, example_args=(inputs["input_ids"],), example_kwargs={"return_dict": True})

# Create pipeline stage for each rank
torch.distributed.init_process_group(rank=rank, world_size=world_size)
stage = PipelineStage(llama_pipe, rank, device=device)


# The `DotDict` class adds dot notation access to dictionary attributes.
class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

def stringify(obj):
    if isinstance(obj, torch.Tensor):
        return str(obj.shape)
    elif isinstance(obj, (list, tuple)):
        return [stringify(o) for o in obj]
    elif hasattr(obj, "items"):
        return {k: stringify(v) for k, v in obj.items()}
    else:
        return str(obj)


# This is an experimental utility function that replaces the original model's forward method with PiPPy's PipelineDriver
# forward method.  It is used to support HuggingFace's `generate()` method, which is defined in a `GenerationMixin`
# class that `PreTrainedModel` inherits from.  We choose this replacement path instead of writing our own `generate()`
# method because the `generate()` method would call into many `GenerationMixin` APIs that may be implemented differently
# by each model.
def inject_pipeline_forward(
    model: torch.nn.Module,
    stage,
    rank,
):
    # logging.info(
    #     f"Inserting PiPPy pipeline forward into model {model._get_name()}"
    # )
    # Inject pipeline driver as a member object of original model
    setattr(model, "stage", stage)

    # Define a new forward method that uses PiPPy's pipeline driver
    def pippy_forward(self, *args, **kwargs):
        keep_running = torch.tensor([1], device=device)
        dist.broadcast(keep_running, src=0)
        # print(f"{rank=}: {keep_running=}")
        # print(f"{rank=}: {kwargs['input_ids']=}")
        print(f"{stringify(args)=}")
        print(f"{stringify(kwargs)=}")
        output = self.stage(kwargs["input_ids"])

        # print(f"{rank=}: {output=}")
        if output is None:
            # print(f"{rank=}: RECEIVING RESULT")
            output = 33*[None,]
            dist.broadcast_object_list(output, src=world_size-1)
        # print(f"{rank=}: {output[0]=}")

        output = {"logits": output[0].to(device), "past_key_values": tuple(o.to(device) for o in output[1:])}

        print(stringify(output))

        if isinstance(output, dict):
            # Add dot access if output is a dictionary. The output of a traced HF model is a traditional dict which has
            # only [`key`] access.  The wrapping is needed for Transformer versons >= 4.28 which access attributes of
            # output via dot notation, such as `output.logits`.  See for example the `generate()` method and
            # `modeling_output.py`.
            output = DotDict(output)
        return output

    # Replace the forward method in original model
    setattr(model, "forward", types.MethodType(pippy_forward, model))


inject_pipeline_forward(llama, stage, rank)

output = None
if rank == 0:
    output = llama.generate(
        inputs["input_ids"],
        max_length=10,
    )
    keep_running = torch.tensor([0], device=device)
    # print(f"{rank=}: {keep_running=}")
    dist.broadcast(keep_running, src=0)
else:
    keep_running = torch.tensor([0], device=device)
    dist.broadcast(keep_running, src=0)
    # print(f"{rank=}: {keep_running=}")
    while keep_running.item() == 1:
        # print(f"{rank=}: ENTERING WHILE")
        output = stage(None)

        if rank != world_size-1:
            output = 33*[None,]
        else:
            # print(f"{rank=}: SENDING RESULT {type(output)}")
            output = list(output)
            # print(f"{rank=}: {len(output)}")

        dist.broadcast_object_list(output, src=world_size-1)
        output = None

        dist.broadcast(keep_running, src=0)
        # print(f"{rank=}: {keep_running=}")


# # Run
# if rank == 0:
#     args = inputs["input_ids"]
# else:
#     args = None
# output = stage(args)

# print(stringify(output))

# Decode
if output is not None:
    # next_token_logits = output[0][:, -1, :]
    # next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(output))
