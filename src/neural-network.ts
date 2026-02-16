// Neural network architecture configuration
export const NETWORK_CONFIG = {
  INPUT_SIZE: 2,
  HIDDEN_SIZE: 32,
  OUTPUT_SIZE: 1,
  LEARNING_RATE: 0.05,
};

// Xavier/He weight initialization for a layer
export function initializeLayerWeights(layer: { input: number; output: number }): Float32Array {
  const numWeights = layer.input * layer.output;
  const packedWeights = new Float32Array(numWeights);
  const stddev = Math.sqrt(2.0 / layer.input);

  for (let i = 0; i < numWeights; i++) {
    // Random normal distribution (Box-Muller transform)
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    packedWeights[i] = z * stddev;
  }

  return packedWeights;
}

// Initialize all neural network weights
export function initializeWeights() {
  const weights1 = initializeLayerWeights({
    input: NETWORK_CONFIG.INPUT_SIZE,
    output: NETWORK_CONFIG.HIDDEN_SIZE,
  });
  const bias1 = new Float32Array(NETWORK_CONFIG.HIDDEN_SIZE);
  
  const weights2 = initializeLayerWeights({
    input: NETWORK_CONFIG.HIDDEN_SIZE,
    output: NETWORK_CONFIG.OUTPUT_SIZE,
  });
  const bias2 = new Float32Array(NETWORK_CONFIG.OUTPUT_SIZE);

  return { weights1, bias1, weights2, bias2 };
}
