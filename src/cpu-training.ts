// CPU version of the NN inference and training

import { NETWORK_CONFIG, initializeLayerWeights } from "./neural-network.js";
import type { TrainingPoint } from "./data-management.js";

interface ActivationFunction {
  compute(value: number): number;
  derivative(value: number): number;
}

interface LossFunction {
  compute(predicted: number, target: number): number;
  derivative(predicted: number, target: number): number;
}

interface DenseLayer {
  numInputs: number; 
  numUnits: number; 

  weights: Float32Array; // packed 2D array: [numUnits][numInputs]
  bias: Float32Array; // 1D array: [numUnits]

  activation: ActivationFunction;
}

interface NNModel {
  layers: DenseLayer[];
  loss: LossFunction;
}

interface DenseLayerGradients {
  weightGrads: Float32Array; // packed 2D array: [numUnits][numInputs]
  biasGrads: Float32Array; // 1D array: [numUnits]
}

interface ModelGradients {
  layerGradients: DenseLayerGradients[];  // per layer
}

interface ForwardPassResults {
  outputs: Float32Array[];  // per sample in the batch
  preActivations: Float32Array[][]; // per sample, per layer
}

// Activation functions
const sigmoid: ActivationFunction = {
  compute: (x: number) => 1 / (1 + Math.exp(-x)),
  derivative: (x: number) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 - s);
  }
};

const relu: ActivationFunction = {
  compute: (x: number) => Math.max(0, x),
  derivative: (x: number) => x > 0 ? 1 : 0
};

// Loss function
const binaryCrossEntropy: LossFunction = {
  compute: (predicted: number, target: number) => {
    const epsilon = 1e-7;
    const p = Math.max(epsilon, Math.min(1 - epsilon, predicted));
    return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
  },
  derivative: (predicted: number, target: number) => {
    const epsilon = 1e-7;
    const p = Math.max(epsilon, Math.min(1 - epsilon, predicted));
    return (p - target) / (p * (1 - p));
  }
};


function forwardPass(model: NNModel, inputsBatch: Float32Array[]): ForwardPassResults {
  const batchSize = inputsBatch.length;
  const outputs: Float32Array[] = [];
  const preActivations: Float32Array[][] = [];

  for (let i = 0; i < batchSize; i++) {
    let currentInput = inputsBatch[i];
    const samplePreActivations: Float32Array[] = [];

    for (const layer of model.layers) {
      const output = new Float32Array(layer.numUnits);
      const values = new Float32Array(layer.numUnits);

      for (let j = 0; j < layer.numUnits; j++) {
        let sum = layer.bias[j];
        for (let k = 0; k < layer.numInputs; k++) {
          sum += currentInput[k] * layer.weights[j * layer.numInputs + k];
        }
        values[j] = sum;
        output[j] = layer.activation.compute(sum);
      }
      
      samplePreActivations.push(values);
      currentInput = output;
    }
    
    outputs.push(currentInput);
    preActivations.push(samplePreActivations);
  }

  return { outputs, preActivations };
}

function computeGradients(model: NNModel, inputsBatch: Float32Array[], targetsBatch: Float32Array[]): ModelGradients {
  // Backpropagation logic to compute gradients
  const batchSize = inputsBatch.length;
  const numLayers = model.layers.length;
  const layerGradients: DenseLayerGradients[] = [];
  const batchNorm = 1 / batchSize;

  for (let i = 0; i < numLayers; i++) {
    const layer = model.layers[i];
    layerGradients.push({
      weightGrads: new Float32Array(layer.numUnits * layer.numInputs),
      biasGrads: new Float32Array(layer.numUnits),
    });
  }

  // Forward pass to get outputs and pre-activations
  const { outputs, preActivations } = forwardPass(model, inputsBatch);
  
  for (let sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
    // Compute deltas for each layer (backwards)
    
    // Output layer delta
    const outputLayer = model.layers[numLayers - 1];
    let delta = new Float32Array(outputLayer.numUnits);
    for (let j = 0; j < outputLayer.numUnits; j++) {
      const lossGrad = model.loss.derivative(outputs[sampleIdx][j], targetsBatch[sampleIdx][j]);
      const activationGrad = outputLayer.activation.derivative(preActivations[sampleIdx][numLayers - 1][j]);
      delta[j] = lossGrad * activationGrad;
    }
    
    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = model.layers[l];

      let layerInput: Float32Array;
      if (l === 0) {
        layerInput = inputsBatch[sampleIdx];
      } else {
        const prevLayer = model.layers[l - 1];
        layerInput = new Float32Array(prevLayer.numUnits);
        for (let j = 0; j < prevLayer.numUnits; j++) {
          layerInput[j] = prevLayer.activation.compute(preActivations[sampleIdx][l - 1][j]);
        }
      }

      const prevDelta = new Float32Array(layer.numInputs);
      for (let k = 0; k < layer.numInputs; k++) {
        for (let j = 0; j < layer.numUnits; j++) {
          layerGradients[l].weightGrads[j * layer.numInputs + k] += delta[j] * layerInput[k] * batchNorm;
          layerGradients[l].biasGrads[j] += delta[j] * batchNorm;
          prevDelta[k] += delta[j] * layer.weights[j * layer.numInputs + k];
        }
        if (l > 0) {
          prevDelta[k] *= model.layers[l - 1].activation.derivative(preActivations[sampleIdx][l - 1][k]);
        }        
      }
      delta = prevDelta;
    }
  }

  return { layerGradients };
}

function updateWeights(model: NNModel, gredients: ModelGradients, learningRate: number): void {
  // Weight update logic using computed gradients
  for (let i = 0; i < model.layers.length; i++) {
    const layer = model.layers[i];
    const layerGradients = gredients.layerGradients[i];

    for (let j = 0; j < layer.numUnits; j++) {
      for (let k = 0; k < layer.numInputs; k++) {
        layer.weights[j * layer.numInputs + k] -= learningRate * layerGradients.weightGrads[j * layer.numInputs + k];
      }
      layer.bias[j] -= learningRate * layerGradients.biasGrads[j];
    }
  }
}

// Create and initialize the model
export function createModel(): NNModel {
  // const { weights1, bias1, weights2, bias2 } = initializeWeights();
  return {
    layers: [
      {
        numInputs: NETWORK_CONFIG.INPUT_SIZE,
        numUnits: NETWORK_CONFIG.HIDDEN_SIZE,
        weights: initializeLayerWeights({input: NETWORK_CONFIG.INPUT_SIZE, output: NETWORK_CONFIG.HIDDEN_SIZE}),
        bias: new Float32Array(NETWORK_CONFIG.HIDDEN_SIZE),
        activation: relu
      },
      {
        numInputs: NETWORK_CONFIG.HIDDEN_SIZE,
        numUnits: NETWORK_CONFIG.HIDDEN_SIZE,
        weights: initializeLayerWeights({input: NETWORK_CONFIG.HIDDEN_SIZE, output: NETWORK_CONFIG.HIDDEN_SIZE}),
        bias: new Float32Array(NETWORK_CONFIG.HIDDEN_SIZE),
        activation: relu
      },
      {
        numInputs: NETWORK_CONFIG.HIDDEN_SIZE,
        numUnits: NETWORK_CONFIG.OUTPUT_SIZE,
        weights: initializeLayerWeights({input: NETWORK_CONFIG.HIDDEN_SIZE, output: NETWORK_CONFIG.OUTPUT_SIZE}),
        bias: new Float32Array(NETWORK_CONFIG.OUTPUT_SIZE),
        activation: sigmoid
      }
    ],
    loss: binaryCrossEntropy
  };
}

// Calculate loss and accuracy
function calculateMetrics(model: NNModel, inputs: Float32Array[], targets: Float32Array[]): { loss: number; accuracy: number } {
  const { outputs } = forwardPass(model, inputs);
  let totalLoss = 0;
  let correct = 0;
  
  for (let i = 0; i < outputs.length; i++) {
    const predicted = outputs[i][0];
    const target = targets[i][0];
    totalLoss += model.loss.compute(predicted, target);
    
    const predictedClass = predicted > 0.5 ? 1 : 0;
    const targetClass = target > 0.5 ? 1 : 0;
    if (predictedClass === targetClass) {
      correct++;
    }
  }
  
  return {
    loss: totalLoss / outputs.length,
    accuracy: correct / outputs.length * 100 // in percentage
  };
}

// Main training loop
export async function trainCPU(
  model: NNModel,
  trainingData: TrainingPoint[],
  epochs: number,
  onProgress: (epoch: number, model: NNModel, loss: number, accuracy: number, time: number) => void,
  shouldStop: () => boolean
): Promise<NNModel> {
  // Prepare training data
  const inputs: Float32Array[] = trainingData.map(p => new Float32Array([p.x, p.y]));
  const targets: Float32Array[] = trainingData.map(p => new Float32Array([p.class]));
  
  for (let epoch = 0; epoch < epochs; epoch++) {
    if (shouldStop()) break;
    
    const startTime = performance.now();
    
    // Compute gradients
    const gradients = computeGradients(model, inputs, targets);
    
    // Update weights
    updateWeights(model, gradients, NETWORK_CONFIG.LEARNING_RATE);
    
    // Calculate metrics every 10 epochs
    if (epoch % 10 === 0) {
      const { loss, accuracy } = calculateMetrics(model, inputs, targets);
      const endTime = performance.now();
      const epochTime = endTime - startTime;
      
      onProgress(epoch, model, loss, accuracy, epochTime);
      await new Promise(resolve => setTimeout(resolve, 0)); // Yield to UI
    }
  }
  
  return model;
}

// Predict for visualization
export function predictCPU(model: NNModel, x: number, y: number): number {
  const input = [new Float32Array([x, y])];
  const { outputs } = forwardPass(model, input);
  return outputs[0][0];
}
