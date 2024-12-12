import torch.nn as nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node


@add_converter(operation_type='LessOrEqual', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    class TorchLessOrEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x0, x1):
            return x0 <= x1

    return OperationConverterResult(
        torch_module=TorchLessOrEqual(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
