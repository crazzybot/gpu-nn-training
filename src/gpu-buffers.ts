import { NETWORK_CONFIG } from "./neural-network.js";

const { INPUT_SIZE, HIDDEN_SIZE } = NETWORK_CONFIG;

export interface GPUBuffers {
  inputBuffer: GPUBuffer;
  targetBuffer: GPUBuffer;
  weights1Buffer: GPUBuffer;
  bias1Buffer: GPUBuffer;
  weights2Buffer: GPUBuffer;
  bias2Buffer: GPUBuffer;
  preActivationBuffer: GPUBuffer;
  outputBuffer: GPUBuffer;
  outputReadBuffer: GPUBuffer;
  gradWeights1Buffer: GPUBuffer;
  gradBias1Buffer: GPUBuffer;
  gradWeights2Buffer: GPUBuffer;
  gradBias2Buffer: GPUBuffer;
  batchSize: number;
}

// Create GPU buffers for neural network training
export function createBuffers(
  device: GPUDevice,
  data: {
    inputs: Float32Array;
    targets: Float32Array;
    weights1: Float32Array;
    bias1: Float32Array;
    weights2: Float32Array;
    bias2: Float32Array;
  }
): GPUBuffers {
  const batchSize = data.targets.length;

  const inputBuffer = device.createBuffer({
    size: data.inputs.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, data.inputs);

  const targetBuffer = device.createBuffer({
    size: data.targets.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(targetBuffer, 0, data.targets);

  const weights1Buffer = device.createBuffer({
    size: data.weights1.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(weights1Buffer, 0, data.weights1);

  const bias1Buffer = device.createBuffer({
    size: data.bias1.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(bias1Buffer, 0, data.bias1);

  const weights2Buffer = device.createBuffer({
    size: data.weights2.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(weights2Buffer, 0, data.weights2);

  const bias2Buffer = device.createBuffer({
    size: data.bias2.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(bias2Buffer, 0, data.bias2);

  const preActivationBuffer = device.createBuffer({
    size: batchSize * (HIDDEN_SIZE + 1) * 4 ,
    usage: GPUBufferUsage.STORAGE,
  });

  const outputBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const outputReadBuffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Gradient buffers (one per sample in batch)
  const gradWeights1Buffer = device.createBuffer({
    size: batchSize * INPUT_SIZE * HIDDEN_SIZE * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const gradBias1Buffer = device.createBuffer({
    size: batchSize * HIDDEN_SIZE * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const gradWeights2Buffer = device.createBuffer({
    size: batchSize * HIDDEN_SIZE * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const gradBias2Buffer = device.createBuffer({
    size: batchSize * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  return {
    inputBuffer,
    targetBuffer,
    weights1Buffer,
    bias1Buffer,
    weights2Buffer,
    bias2Buffer,
    preActivationBuffer,
    outputBuffer,
    outputReadBuffer,
    gradWeights1Buffer,
    gradBias1Buffer,
    gradWeights2Buffer,
    gradBias2Buffer,
    batchSize,
  };
}
