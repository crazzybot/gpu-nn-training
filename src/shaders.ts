import { NETWORK_CONFIG } from "./neural-network.js";

const { INPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE } = NETWORK_CONFIG;

// WGSL shader for forward pass
export const forwardShader = `
  struct Matrix {
      data: array<f32>
  }

  @group(0) @binding(0) var<storage, read> input: Matrix;
  @group(0) @binding(1) var<storage, read> weights1: Matrix;
  @group(0) @binding(2) var<storage, read> bias1: Matrix;
  @group(0) @binding(3) var<storage, read> weights2: Matrix;
  @group(0) @binding(4) var<storage, read> bias2: Matrix;
  @group(0) @binding(5) var<storage, read_write> pre_activation: Matrix; // packed [hidden_size][output_size]
  @group(0) @binding(6) var<storage, read_write> output: Matrix;

  fn relu(x: f32) -> f32 {
      return max(0.0, x);
  }

  fn sigmoid(x: f32) -> f32 {
      return 1.0 / (1.0 + exp(-x));
  }

  @compute @workgroup_size(1)
  fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      let batch_size = arrayLength(&input.data) / ${INPUT_SIZE}u;
      
      if (idx >= batch_size) {
          return;
      }

      let pre_activation_offset = idx * (${HIDDEN_SIZE}u + 1);

      // Layer 1: input -> hidden
      // Layer 2: hidden -> output
      var out_sum = bias2.data[0];
      for (var h = 0u; h < ${HIDDEN_SIZE}u; h++) {
          var sum = bias1.data[h];
          for (var i = 0u; i < ${INPUT_SIZE}u; i++) {
              sum += input.data[idx * ${INPUT_SIZE}u + i] * weights1.data[i * ${HIDDEN_SIZE}u + h];
          }
          pre_activation.data[pre_activation_offset + h] = sum;
          out_sum += relu(sum) * weights2.data[h];
      }

      pre_activation.data[pre_activation_offset + ${HIDDEN_SIZE}u] = out_sum;
      output.data[idx] = sigmoid(out_sum);
  }
`;

// WGSL shader for computing gradients
export const gradientShader = `
  struct Matrix {
      data: array<f32>
  }

  @group(0) @binding(0) var<storage, read> input: Matrix;
  @group(0) @binding(1) var<storage, read> pre_activation: Matrix;
  @group(0) @binding(2) var<storage, read> output: Matrix;
  @group(0) @binding(3) var<storage, read> target_val: Matrix;
  @group(0) @binding(4) var<storage, read> weights2: Matrix;
  @group(0) @binding(5) var<storage, read_write> grad_weights1: Matrix;
  @group(0) @binding(6) var<storage, read_write> grad_bias1: Matrix;
  @group(0) @binding(7) var<storage, read_write> grad_weights2: Matrix;
  @group(0) @binding(8) var<storage, read_write> grad_bias2: Matrix;
  
  fn sigmoid_derivative(x: f32) -> f32 {
      let s = 1.0 / (1.0 + exp(-x));
      return s * (1.0 - s);
  }

  fn relu_derivative(x: f32) -> f32 {
      return select(0.0, 1.0, x > 0.0);
  }

  fn relu(x: f32) -> f32 {
      return max(0.0, x);
  }

  fn sigmoid(x: f32) -> f32 {
      return 1.0 / (1.0 + exp(-x));
  }

  @compute @workgroup_size(1)
  fn compute_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      let batch_size = arrayLength(&input.data) / ${INPUT_SIZE}u;
      
      if (idx >= batch_size) {
          return;
      }

      let pre_activation_offset = idx * (${HIDDEN_SIZE}u + 1);

      // Output layer gradient
      let output_delta = (output.data[idx] - target_val.data[idx]);

      // Store gradients for weights2 and bias2
      var deltas = array<f32, ${HIDDEN_SIZE}u>();
      for (var h = 0u; h < ${HIDDEN_SIZE}u; h++) {
          deltas[h] = output_delta * weights2.data[h] * relu_derivative(pre_activation.data[pre_activation_offset + h]);
          grad_weights2.data[idx * ${HIDDEN_SIZE}u + h] = relu(pre_activation.data[pre_activation_offset + h]) * output_delta;
      }
      grad_bias2.data[idx] = output_delta;

      // Hidden layer gradients
      for (var i = 0u; i < ${INPUT_SIZE}u; i++) {
        for (var h = 0u; h < ${HIDDEN_SIZE}u; h++) {
          let grad = deltas[h] * input.data[idx * ${INPUT_SIZE}u + i];
          grad_weights1.data[idx * ${INPUT_SIZE}u * ${HIDDEN_SIZE}u + i * ${HIDDEN_SIZE}u + h] = grad;
        }
      }

      for (var h = 0u; h < ${HIDDEN_SIZE}u; h++) {
        grad_bias1.data[idx * ${HIDDEN_SIZE}u + h] = deltas[h];
      }
  }
`;

// WGSL shader for applying gradients
export const updateWeightsShader = `
  struct Matrix {
      data: array<f32>
  }

  @group(0) @binding(0) var<storage, read> grad_weights1: Matrix;
  @group(0) @binding(1) var<storage, read> grad_bias1: Matrix;
  @group(0) @binding(2) var<storage, read> grad_weights2: Matrix;
  @group(0) @binding(3) var<storage, read> grad_bias2: Matrix;
  @group(0) @binding(4) var<storage, read_write> weights1: Matrix;
  @group(0) @binding(5) var<storage, read_write> bias1: Matrix;
  @group(0) @binding(6) var<storage, read_write> weights2: Matrix;
  @group(0) @binding(7) var<storage, read_write> bias2: Matrix;

  @compute @workgroup_size(1)
  fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let param_idx = global_id.x;
      let batch_size = arrayLength(&grad_bias2.data);
      let lr = ${LEARNING_RATE};

      // Update weights1
      let total_weights1 = ${INPUT_SIZE}u * ${HIDDEN_SIZE}u;
      if (param_idx < total_weights1) {
          var grad_sum = 0.0;
          for (var b = 0u; b < batch_size; b++) {
              grad_sum += grad_weights1.data[b * total_weights1 + param_idx];
          }
          weights1.data[param_idx] -= lr * grad_sum / f32(batch_size);
          return;
      }

      // Update bias1
      let bias1_idx = param_idx - total_weights1;
      if (bias1_idx < ${HIDDEN_SIZE}u) {
          var grad_sum = 0.0;
          for (var b = 0u; b < batch_size; b++) {
              grad_sum += grad_bias1.data[b * ${HIDDEN_SIZE}u + bias1_idx];
          }
          bias1.data[bias1_idx] -= lr * grad_sum / f32(batch_size);
          return;
      }

      // Update weights2
      let weights2_idx = bias1_idx - ${HIDDEN_SIZE}u;
      if (weights2_idx < ${HIDDEN_SIZE}u) {
          var grad_sum = 0.0;
          for (var b = 0u; b < batch_size; b++) {
              grad_sum += grad_weights2.data[b * ${HIDDEN_SIZE}u + weights2_idx];
          }
          weights2.data[weights2_idx] -= lr * grad_sum / f32(batch_size);
          return;
      }

      // Update bias2
      let bias2_idx = weights2_idx - ${HIDDEN_SIZE}u;
      if (bias2_idx == 0u) {
          var grad_sum = 0.0;
          for (var b = 0u; b < batch_size; b++) {
              grad_sum += grad_bias2.data[b];
          }
          bias2.data[0] -= lr * grad_sum / f32(batch_size);
      }
  }
`;
