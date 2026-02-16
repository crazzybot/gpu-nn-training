import { NETWORK_CONFIG } from "./neural-network.js";
import type { TrainingPoint } from "./data-management.js";

const { HIDDEN_SIZE, INPUT_SIZE } = NETWORK_CONFIG;

// Draw training data points on a canvas
export function drawTrainingData(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  trainingData: TrainingPoint[]
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  trainingData.forEach((point) => {
    ctx.fillStyle = point.class === 1 ? "#6495ED" : "#DC143C";
    ctx.beginPath();
    ctx.arc(
      point.x * canvas.width,
      point.y * canvas.height,
      8,
      0,
      Math.PI * 2
    );
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

// Visualize decision boundary on the model canvas
export async function visualizeModel(
  device: GPUDevice,
  buffers: {
    weights1Buffer: GPUBuffer;
    bias1Buffer: GPUBuffer;
    weights2Buffer: GPUBuffer;
    bias2Buffer: GPUBuffer;
  },
  pipeline: GPUComputePipeline,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  trainingData: TrainingPoint[]
) {
  const resolution = 50;
  const gridSize = resolution * resolution;
  const gridInputs = new Float32Array(gridSize * INPUT_SIZE);

  let idx = 0;
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      gridInputs[idx++] = i / resolution;
      gridInputs[idx++] = j / resolution;
    }
  }

  const gridInputBuffer = device.createBuffer({
    size: gridInputs.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridInputBuffer, 0, gridInputs);

  const gridOutputBuffer = device.createBuffer({
    size: gridSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const gridPreActivationBuffer = device.createBuffer({
    size: gridSize * (HIDDEN_SIZE + 1) * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const gridReadBuffer = device.createBuffer({
    size: gridSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const gridBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gridInputBuffer } },
      { binding: 1, resource: { buffer: buffers.weights1Buffer } },
      { binding: 2, resource: { buffer: buffers.bias1Buffer } },
      { binding: 3, resource: { buffer: buffers.weights2Buffer } },
      { binding: 4, resource: { buffer: buffers.bias2Buffer } },
      { binding: 5, resource: { buffer: gridPreActivationBuffer } },
      { binding: 6, resource: { buffer: gridOutputBuffer } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, gridBindGroup);
  computePass.dispatchWorkgroups(gridSize);
  computePass.end();

  commandEncoder.copyBufferToBuffer(
    gridOutputBuffer,
    0,
    gridReadBuffer,
    0,
    gridSize * 4
  );
  device.queue.submit([commandEncoder.finish()]);

  await gridReadBuffer.mapAsync(GPUMapMode.READ);
  const predictions = new Float32Array(gridReadBuffer.getMappedRange());

  // Draw decision boundary
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cellSize = canvas.width / resolution;

  idx = 0;
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const value = predictions[idx++];
      const alpha = Math.abs(value - 0.5) * 2;
      ctx.fillStyle =
        value > 0.5
          ? `rgba(100, 149, 237, ${alpha * 0.3})`
          : `rgba(220, 20, 60, ${alpha * 0.3})`;
      ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
    }
  }

  // Draw training points
  trainingData.forEach((point) => {
    ctx.fillStyle = point.class === 1 ? "#6495ED" : "#DC143C";
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  gridReadBuffer.unmap();
}

// Draw loss chart
export function drawLossChart(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  lossHistory: number[]
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory);
  const minLoss = Math.min(...lossHistory);
  const range = maxLoss - minLoss || 1;

  ctx.strokeStyle = "#667eea";
  ctx.lineWidth = 2;
  ctx.beginPath();

  lossHistory.forEach((loss, i) => {
    const x = (i / (lossHistory.length - 1)) * canvas.width;
    const y =
      canvas.height - ((loss - minLoss) / range) * (canvas.height - 20) - 10;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();
}
// Visualize CPU model decision boundary
export function visualizeModelCPU(
  predictFn: (x: number, y: number) => number,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  trainingData: TrainingPoint[]
) {
  const resolution = 50;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cellSize = canvas.width / resolution;

  // Draw decision boundary
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const x = i / resolution;
      const y = j / resolution;
      const value = predictFn(x, y);
      const alpha = Math.abs(value - 0.5) * 2;
      ctx.fillStyle =
        value > 0.5
          ? `rgba(100, 149, 237, ${alpha * 0.3})`
          : `rgba(220, 20, 60, ${alpha * 0.3})`;
      ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
    }
  }

  // Draw training points
  trainingData.forEach((point) => {
    ctx.fillStyle = point.class === 1 ? "#6495ED" : "#DC143C";
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}