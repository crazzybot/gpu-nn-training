import { initWebGPU } from "./webgpu.js";
import { NETWORK_CONFIG } from "./neural-network.js";
import { prepareTrainingData, generateSpiral, generateCircle, type TrainingPoint } from "./data-management.js";
import { createTrainingPipelines, calculateLoss } from "./gpu-training.js";
import { createModel, trainCPU, predictCPU } from "./cpu-training.js";
import { drawTrainingData, visualizeModel, visualizeModelCPU, drawLossChart } from "./visualization.js";
import { updateStatus, updateMetrics, clearMetrics, setButtonStates } from "./ui-handlers.js";

const { INPUT_SIZE, HIDDEN_SIZE } = NETWORK_CONFIG;

// Initial draw
window.addEventListener("DOMContentLoaded", () => {
  // Check WebGPU support
  if (!navigator.gpu) {
    updateStatus(
      "WebGPU is not supported in your browser. Please use Chrome/Edge 113+ or update your browser.",
      "error"
    );
    setButtonStates({ trainBtn: true });
    return;
  }

  // Training data
  let trainingData: TrainingPoint[] = [];
  let currentClass = 0;
  let isTraining = false;
  let device: GPUDevice;
  let lossHistory: number[] = [];
  let cpuModel: any = null;

  // Canvas setup
  const dataCanvas = document.getElementById("dataCanvas") as HTMLCanvasElement;
  const dataCtx = dataCanvas!.getContext("2d")!;
  const modelCanvas = document.getElementById("modelCanvas") as HTMLCanvasElement;
  const modelCtx = modelCanvas!.getContext("2d")!;
  const lossChart = document.getElementById("lossChart") as HTMLCanvasElement;
  const lossCtx = lossChart!.getContext("2d")!;

  // Train the model
  async function trainModel() {
    if (trainingData.length < 2) {
      updateStatus("Please add at least 2 training points!", "error");
      return;
    }

    // Get selected training mode
    const trainingMode = (document.querySelector('input[name="trainingMode"]:checked') as HTMLInputElement)?.value || 'gpu';

    isTraining = true;
    setButtonStates({ trainBtn: true, stopBtn: false });
    lossHistory = [];

    try {
      if (trainingMode === 'cpu') {
        // CPU Training Mode
        updateStatus("Training on CPU...", "info");

        let lastVisualizeEpoch = 0;
        const model = createModel();

        cpuModel = await trainCPU(
          model,
          trainingData,
          100000,
          (epoch, model, loss, accuracy, time) => {
            updateMetrics({ epoch, loss, accuracy, time });
            lossHistory.push(loss);
            drawLossChart(lossChart, lossCtx, lossHistory);

            // Visualize decision boundary every 50 epochs
            if (epoch - lastVisualizeEpoch >= 50) {
              visualizeModelCPU(
                (x, y) => predictCPU(model, x, y),
                modelCanvas,
                modelCtx,
                trainingData
              );
              lastVisualizeEpoch = epoch;
            }
          },
          () => !isTraining
        );

        if (isTraining) {
          updateStatus("Training completed!", "success");
          // Final visualization
          visualizeModelCPU(
            (x, y) => predictCPU(cpuModel, x, y),
            modelCanvas,
            modelCtx,
            trainingData
          );
        }
      } else {
        // GPU Training Mode
        updateStatus("Initializing WebGPU...", "info");

        if (!device) {
          device = await initWebGPU();
        }

        // Prepare data
        const { inputs, targets } = prepareTrainingData(trainingData);

        // Create compute pipelines
        const pipelines = createTrainingPipelines(device, inputs, targets);

        updateStatus("Training started...", "success");

        // Training loop
        for (let epoch = 0; epoch < 100000 && isTraining; epoch++) {
          const startTime = performance.now();
          const commandEncoder = device.createCommandEncoder();

          // Forward pass
          const forwardPass = commandEncoder.beginComputePass();
          forwardPass.setPipeline(pipelines.forwardPipeline);
          forwardPass.setBindGroup(0, pipelines.forwardBindGroup);
          forwardPass.dispatchWorkgroups(pipelines.buffers.batchSize);
          forwardPass.end();

          // Update gradient pass
          const backwardPass = commandEncoder.beginComputePass();
          backwardPass.setPipeline(pipelines.gradientPipeline);
          backwardPass.setBindGroup(0, pipelines.gradientBindGroup);
          backwardPass.dispatchWorkgroups(pipelines.buffers.batchSize);
          backwardPass.end();

          // Update weights pass
          const updatePass = commandEncoder.beginComputePass();
          updatePass.setPipeline(pipelines.updateWeightsPipeline);
          updatePass.setBindGroup(0, pipelines.updateWeightsBindGroup);
          updatePass.dispatchWorkgroups((INPUT_SIZE + 2) * HIDDEN_SIZE + 1);
          updatePass.end();

          // Copy output for loss calculation
          if (epoch % 10 === 0) {
            commandEncoder.copyBufferToBuffer(
              pipelines.buffers.outputBuffer,
              0,
              pipelines.buffers.outputReadBuffer,
              0,
              pipelines.buffers.batchSize * 4
            );
          }

          device.queue.submit([commandEncoder.finish()]);

          // Calculate loss every 10 epochs
          if (epoch % 10 === 0) {
            await pipelines.buffers.outputReadBuffer.mapAsync(GPUMapMode.READ);
            const outputData = new Float32Array(
              pipelines.buffers.outputReadBuffer.getMappedRange()
            );

            const { loss, accuracy } = calculateLoss(outputData, targets);
            pipelines.buffers.outputReadBuffer.unmap();

            const endTime = performance.now();
            const epochTime = endTime - startTime;

            // Update UI
            updateMetrics({ epoch, loss, accuracy, time: epochTime });

            lossHistory.push(loss);
            drawLossChart(lossChart, lossCtx, lossHistory);

            // Visualize decision boundary
            if (epoch % 50 === 0) {
              await visualizeModel(
                device,
                pipelines.buffers,
                pipelines.forwardPipeline,
                modelCanvas,
                modelCtx,
                trainingData
              );
            }
          }

          // Allow UI to update
          if (epoch % 10 === 0) {
            await new Promise((resolve) => setTimeout(resolve, 0));
          }
        }

        if (isTraining) {
          updateStatus("Training completed!", "success");
        }
      }
    } catch (error) {
      console.error("Training error:", error);
      updateStatus("Training failed: " + (error as Error).message, "error");
    }

    isTraining = false;
    setButtonStates({ trainBtn: false, stopBtn: true });
  }

  // Canvas click handler
  dataCanvas.addEventListener("click", (e) => {
    const rect = dataCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / dataCanvas.width;
    const y = (e.clientY - rect.top) / dataCanvas.height;

    trainingData.push({ x, y, class: currentClass });
    currentClass = 1 - currentClass;

    drawTrainingData(dataCanvas, dataCtx, trainingData);
    updateStatus(
      `${trainingData.length} training points added. Click "Start Training" when ready!`,
      "info"
    );
  });

  // Clear data
  function clearData() {
    trainingData = [];
    currentClass = 0;
    lossHistory = [];
    cpuModel = null;
    dataCtx.clearRect(0, 0, dataCanvas.width, dataCanvas.height);
    modelCtx.clearRect(0, 0, modelCanvas.width, modelCanvas.height);
    lossCtx.clearRect(0, 0, lossChart.width, lossChart.height);
    clearMetrics();
    updateStatus(
      "Data cleared. Click on the canvas to add new training points!",
      "info"
    );
  }

  // Generate datasets
  function loadSpiral() {
    clearData();
    trainingData = generateSpiral();
    drawTrainingData(dataCanvas, dataCtx, trainingData);
  }

  function loadCircle() {
    clearData();
    trainingData = generateCircle();
    drawTrainingData(dataCanvas, dataCtx, trainingData);
  }

  // Event listeners
  document.getElementById("trainBtn")!.addEventListener("click", trainModel);
  document.getElementById("stopBtn")!.addEventListener("click", () => {
    isTraining = false;
    updateStatus("Training stopped by user", "info");
  });

  // Attach clear and generate buttons
  document.getElementById("clear-btn")!.addEventListener("click", clearData);
  document
    .getElementById("generate-spiral-btn")!
    .addEventListener("click", loadSpiral);
  document
    .getElementById("generate-circle-btn")!
    .addEventListener("click", loadCircle);

  // Initial draw
  drawTrainingData(dataCanvas, dataCtx, trainingData);
});
