import torch.nn as nn
import torch.nn.functional as F

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node


@add_converter(operation_type='GridSample', version=16)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    class TorchGridSample(nn.Module):
        def __init__(self, mode, padding_mode, align_corners):
            super().__init__()
            self.mode = mode
            self.padding_mode = padding_mode
            self.align_corners = align_corners

        def forward(self, x, grid):
            return F.grid_sample(x, grid, self.mode, self.padding_mode, self.align_corners)

    node_attributes = node.attributes
    mode = node_attributes.get('mode', 'linear')
    padding_mode = node_attributes.get('padding_mode', 'zeros')
    align_corners = node_attributes.get('align_corners', False)
    return OperationConverterResult(
        torch_module=TorchGridSample(mode, padding_mode, bool(align_corners)),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
