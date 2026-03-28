# version 2. To use this

import numpy as np

FRAC_BITS   = 16
FRAC_SCALE  = 1 << FRAC_BITS
INT32_MAX   = (1<<31) - 1
INT32_MIN   = -(1<<31)

def saturate_int32(x: np.ndarray) -> np.ndarray:
    return np.clip(x, INT32_MIN, INT32_MAX).astype(np.int32)

def requantize(x: np.ndarray) -> np.ndarray:
    x_shifted = x >> FRAC_BITS
    return saturate_int32(x_shifted)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x) # elementwise

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) / FRAC_SCALE

    # to prevent overflow as each element of x <= 0
    # 1 element is guranteed = 0, hence exp(0) = 1 will prevent undeflow
    x = x - np.max(x)

    x = np.exp(x)
    return x / np.sum(x)

def fcl_single_layer(weights: np.ndarray,
                     bias: np.ndarray,
                     data: np.ndarray,
                     input_size: int,
                     output_size: int,
                     apply_relu: bool = True) -> np.ndarray:
    if len(weights) != output_size*input_size:
        raise ValueError(
            f'Number of weight {len(weights)} != required number of weights'
            f'Required weights = output_size * input_size = {output_size} * {input_size} = {output_size*input_size}'
        )
    if len(bias) != output_size:
        raise ValueError(
            f'Bias {len(bias)} != Required number of biases'
            f'Required bias = {output_size}'
        )
    if len(data) != input_size:
        raise ValueError(
            f'Data size {len(data)} != input size {input_size}'
        )
    
    W = weights.reshape(output_size, input_size).astype(np.int64)
    d = data.astype(np.int64)
    b = bias.astype(np.int64)

    # accumulated sum = Matrix Mul (W, d); symbol = @; Python version >= 3.5
    acc = W @ d

    acc = requantize(acc)
    acc = acc.astype(np.int64)+b
    acc = saturate_int32(acc)

    if apply_relu:
        acc = relu(acc)

    return acc.astype(np.int32)

def fcl_total(control: list,
              weights_total: np.ndarray,
              bias_total: np.ndarray,
              data: np.ndarray) -> np.ndarray:
    if len(control) < 2:
        raise ValueError("control list must have at least 2 elements (one layer)")

    # Validate total weight and bias counts upfront
    expected_weights = sum(control[i] * control[i+1] for i in range(len(control) - 1))
    expected_biases  = sum(control[i+1]               for i in range(len(control) - 1))

    if len(weights_total) != expected_weights:
        raise ValueError(
            f"weights_total length {len(weights_total)} does not match "
            f"expected {expected_weights} for control={control}"
        )
    if len(bias_total) != expected_biases:
        raise ValueError(
            f"bias_total length {len(bias_total)} does not match "
            f"expected {expected_biases} for control={control}"
        )

    output = data.flatten().astype(np.int32)

    weight_offset = 0
    bias_offset   = 0
    num_layers    = len(control) - 1

    for i in range(num_layers):
        input_size  = control[i]
        output_size = control[i + 1]
        is_last     = (i == num_layers - 1)

        w_count = output_size * input_size
        b_count = output_size

        layer_weights = weights_total[weight_offset : weight_offset + w_count]
        layer_bias    = bias_total[bias_offset     : bias_offset   + b_count]

        output = fcl_single_layer(
            weights=layer_weights,
            bias=layer_bias,
            data=output,
            input_size=input_size,
            output_size=output_size,
            apply_relu=not is_last      # No ReLU on final layer
        )

        weight_offset += w_count
        bias_offset   += b_count

    # Final output: convert fixed-point logits to softmax probabilities
    return softmax(output)
