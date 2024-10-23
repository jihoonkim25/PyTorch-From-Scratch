from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Loop over batches, output channels, and output width
    for batch_index in prange(batch_):
        for out_channel in prange(out_channels_):
            for out_position in prange(out_width):
                # Accumulator for the sum of kernel products
                total_acc = 0.0
                for in_channel in prange(in_channels):  # Loop over input channels
                    for kernel_index in prange(kw):  # Loop over kernel width
                        # Calculate position in the weight tensor
                        weight_pos = (
                            out_channel * s2[0]
                            + in_channel * s2[1]
                            + kernel_index * s2[2]
                        )
                        # Initialize index array for input tensor
                        input_index = np.zeros(3, np.int16)
                        # Get position of input tensor for standard conv
                        if not reverse:
                            input_pos = (
                                batch_index * s1[0]
                                + in_channel * s1[1]
                                + out_position * s1[2]
                                + kernel_index * s1[2]
                            )
                            val = out_position + kernel_index
                        else:  # Get position of input tensor for reverse conv
                            input_pos = (
                                batch_index * s1[0]
                                + in_channel * s1[1]
                                + out_position * s1[2]
                                - kernel_index * s1[2]
                            )
                            val = out_position - kernel_index
                        # Set input tensor indices
                        input_index[0] = batch_index
                        input_index[1] = in_channel
                        input_index[2] = val
                        # Perform convolution operation based on the reverse flag and boundary conditions
                        if (input_index[2] < width and not reverse) or (
                            input_index[2] >= 0 and reverse
                        ):
                            total_acc += input[input_pos] * weight[weight_pos]

                # Calculate position in the output tensor and store result
                out_pos = (
                    batch_index * out_strides[0]
                    + out_channel * out_strides[1]
                    + out_position * out_strides[2]
                )
                out[out_pos] = total_acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Loop over each dimension of the output tensor
    for batch_index in prange(batch_):
        for out_channel in prange(out_channels):
            for out_row in prange(out_shape[2]):
                for out_col in prange(out_shape[3]):
                    # Calculate position in the output tensor
                    out_position = (
                        batch_index * out_strides[0]
                        + out_channel * out_strides[1]
                        + out_row * out_strides[2]
                        + out_col * out_strides[3]
                    )

                    # Accumulator for the sum of products
                    total_acc = 0.0

                    # Loop over input channels and kernel dimensions
                    for in_channel in range(in_channels):
                        for kernel_row in range(kh):
                            for kernel_col in range(kw):
                                # Calculate positions in input and weight tensors
                                if not reverse:  # Standard convolution
                                    # Check boundary conditions
                                    if (
                                        out_row + kernel_row < height
                                        and out_col + kernel_col < width
                                    ):
                                        input_pos = (
                                            batch_index * s10
                                            + in_channel * s11
                                            + (out_row + kernel_row) * s12
                                            + (out_col + kernel_col) * s13
                                        )
                                        weight_pos = (
                                            out_channel * s20
                                            + in_channel * s21
                                            + kernel_row * s22
                                            + kernel_col * s23
                                        )
                                        total_acc += (
                                            input[input_pos] * weight[weight_pos]
                                        )
                                else:  # Reverse convolution
                                    # Check boundary conditions
                                    if (out_row - kernel_row) >= 0 and (
                                        out_col - kernel_col
                                    ) >= 0:
                                        input_pos = (
                                            batch_index * s10
                                            + in_channel * s11
                                            + (out_row - kernel_row) * s12
                                            + (out_col - kernel_col) * s13
                                        )
                                        weight_pos = (
                                            out_channel * s20
                                            + in_channel * s21
                                            + kernel_row * s22
                                            + kernel_col * s23
                                        )
                                        total_acc += (
                                            input[input_pos] * weight[weight_pos]
                                        )

                    # Store the accumulated total in the output tensor
                    out[out_position] = total_acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
