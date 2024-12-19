__all__ = [
    'OnnxBinaryMathOperation',
]

from typing import Optional

import torch
from torch import nn
import onnx

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import old_style_broadcast
from onnx2torch.utils.common import onnx_mapping_from_node


_TORCH_FUNCTION_FROM_ONNX_TYPE = {
    'Add': torch.add,
    'Sub': torch.sub,
    'Mul': torch.mul,
    'Div': torch.div,
}


class OnnxBinaryMathOperation(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-docstring
    def __init__(self, math_op_function, broadcast: Optional[int] = None, axis: Optional[int] = None):
        super().__init__()

        self.math_op_function = math_op_function
        self.broadcast = broadcast
        self.axis = axis

    def forward(  # pylint: disable=missing-function-docstring
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        if self.broadcast == 1 and self.axis is not None:
            second = old_style_broadcast(first, second, self.axis)

        return self.math_op_function(first, second)


@add_converter(operation_type='Add', version=1)
@add_converter(operation_type='Add', version=6)
@add_converter(operation_type='Add', version=7)
@add_converter(operation_type='Add', version=13)
@add_converter(operation_type='Add', version=14)
@add_converter(operation_type='Sub', version=1)
@add_converter(operation_type='Sub', version=6)
@add_converter(operation_type='Sub', version=7)
@add_converter(operation_type='Sub', version=13)
@add_converter(operation_type='Sub', version=14)
@add_converter(operation_type='Mul', version=1)
@add_converter(operation_type='Mul', version=6)
@add_converter(operation_type='Mul', version=7)
@add_converter(operation_type='Mul', version=13)
@add_converter(operation_type='Mul', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=OnnxBinaryMathOperation(
            math_op_function=_TORCH_FUNCTION_FROM_ONNX_TYPE[node.operation_type],
            broadcast=node.attributes.get('broadcast', None),
            axis=node.attributes.get('axis', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type='Div', version=1)
@add_converter(operation_type='Div', version=6)
@add_converter(operation_type='Div', version=7)
@add_converter(operation_type='Div', version=13)
@add_converter(operation_type='Div', version=14)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    integer_types = (
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.UINT4,
        onnx.TensorProto.INT4,
    )
    dtypes = [
        graph.value_info.get(x, None) and graph.value_info[x].type or graph.initializers[x].proto.data_type for x in node.input_values]
    all_int = all((isinstance(x, int) and x in integer_types) or (x.HasField('tensor_type')
                  and x.tensor_type.elem_type in integer_types) for x in dtypes)
    if all_int:
        return OperationConverterResult(
            torch_module=OnnxBinaryMathOperation(
                math_op_function=lambda lhs, rhs: torch.div(
                    lhs, rhs, rounding_mode='trunc'),
                broadcast=node.attributes.get('broadcast', None),
                axis=node.attributes.get('axis', None),
            ),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )
    return OperationConverterResult(
        torch_module=OnnxBinaryMathOperation(
            math_op_function=_TORCH_FUNCTION_FROM_ONNX_TYPE[node.operation_type],
            broadcast=node.attributes.get('broadcast', None),
            axis=node.attributes.get('axis', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
