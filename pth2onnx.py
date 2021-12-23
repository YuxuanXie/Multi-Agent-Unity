import pickle
from model.conv2mlp import TorchRNNModel
import torch


checkpoint = "/Users/yuxuan/git/maUnity/results/PPO_gc_env_735f2_00000_0_2021-12-21_14-44-31/checkpoint_000100/checkpoint-100"
states = pickle.loads(pickle.load(open(checkpoint, 'rb'))['worker'])['state']['policy-0']

# alg = PPO(93, (3,3,2), config=config)
# alg.load_model("./400033.pth", cuda=False)
# # model = torch.jit.script(alg.model)
# dummy_input = torch.randn(1, 93, requires_grad=False)

# torch.onnx.export(alg.model, dummy_input, "leanring.onnx", export_params=True, opset_version=9, do_constant_folding=True, input_names = ['vector_observation'], output_names = ['action0', 'action1', 'action2', 'value'],)
