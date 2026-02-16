import { forwardShader, gradientShader, updateWeightsShader } from "./shaders.js";
import { initializeWeights } from "./neural-network.js";
import { createBuffers, type GPUBuffers } from "./gpu-buffers.js";


export interface TrainingPipelines {
  forwardPipeline: GPUComputePipeline;
  gradientPipeline: GPUComputePipeline;
  updateWeightsPipeline: GPUComputePipeline;
  forwardBindGroup: GPUBindGroup;
  gradientBindGroup: GPUBindGroup;
  updateWeightsBindGroup: GPUBindGroup;
  buffers: GPUBuffers;
}

// Create compute pipelines for training
export function createTrainingPipelines(
  device: GPUDevice,
  inputs: Float32Array,
  targets: Float32Array
): TrainingPipelines {
  const forwardPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: forwardShader }),
      entryPoint: "forward",
    },
  });

  const gradientPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: gradientShader }),
      entryPoint: "compute_gradients",
    },
  });

  const updateWeightsPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: updateWeightsShader }),
      entryPoint: "update_weights",
    },
  });

  const weights = initializeWeights();
  const buffers = createBuffers(device, {
    inputs,
    targets,
    ...weights,
  });

  const forwardBindGroup = device.createBindGroup({
    layout: forwardPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.inputBuffer } },
      { binding: 1, resource: { buffer: buffers.weights1Buffer } },
      { binding: 2, resource: { buffer: buffers.bias1Buffer } },
      { binding: 3, resource: { buffer: buffers.weights2Buffer } },
      { binding: 4, resource: { buffer: buffers.bias2Buffer } },
      { binding: 5, resource: { buffer: buffers.preActivationBuffer } },
      { binding: 6, resource: { buffer: buffers.outputBuffer } },
    ],
  });

  const gradientBindGroup = device.createBindGroup({
    layout: gradientPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.inputBuffer } },
      { binding: 1, resource: { buffer: buffers.preActivationBuffer } },
      { binding: 2, resource: { buffer: buffers.outputBuffer } },
      { binding: 3, resource: { buffer: buffers.targetBuffer } },
      { binding: 4, resource: { buffer: buffers.weights2Buffer } },
      { binding: 5, resource: { buffer: buffers.gradWeights1Buffer } },
      { binding: 6, resource: { buffer: buffers.gradBias1Buffer } },
      { binding: 7, resource: { buffer: buffers.gradWeights2Buffer } },
      { binding: 8, resource: { buffer: buffers.gradBias2Buffer } },
    ],
  });

  const updateWeightsBindGroup = device.createBindGroup({
    layout: updateWeightsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.gradWeights1Buffer } },
      { binding: 1, resource: { buffer: buffers.gradBias1Buffer } },
      { binding: 2, resource: { buffer: buffers.gradWeights2Buffer } },
      { binding: 3, resource: { buffer: buffers.gradBias2Buffer } },
      { binding: 4, resource: { buffer: buffers.weights1Buffer } },
      { binding: 5, resource: { buffer: buffers.bias1Buffer } },
      { binding: 6, resource: { buffer: buffers.weights2Buffer } },
      { binding: 7, resource: { buffer: buffers.bias2Buffer } },
    ],
  });

  return {
    forwardPipeline,
    gradientPipeline,
    updateWeightsPipeline,
    forwardBindGroup,
    gradientBindGroup,
    updateWeightsBindGroup,
    buffers,
  };
}

// Calculate loss from model output
export function calculateLoss(
  outputData: Float32Array,
  targets: Float32Array
): { loss: number; accuracy: number } {
  let loss = 0;
  let correct = 0;
  const numSamples = outputData.length;

  for (let i = 0; i < numSamples; i++) {
    const pred = outputData[i];
    const target = targets[i];
    loss +=
      -target * Math.log(pred + 1e-7) -
      (1 - target) * Math.log(1 - pred + 1e-7);
    correct += Math.round(pred) === target ? 1 : 0;
  }

  loss /= numSamples;
  const accuracy = (correct / numSamples) * 100;

  return { loss, accuracy };
}
