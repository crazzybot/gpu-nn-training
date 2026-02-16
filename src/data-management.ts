export interface TrainingPoint {
  x: number;
  y: number;
  class: number;
}

// Generate a spiral dataset
export function generateSpiral(): TrainingPoint[] {
  const points = 100;
  const data: TrainingPoint[] = [];
  
  for (let i = 0; i < points; i++) {
    const r = (i / points) * 0.8;
    const t = ((1.75 * i) / points) * 2 * Math.PI;
    
    data.push({
      x: 0.5 + r * Math.cos(t) * 0.5,
      y: 0.5 + r * Math.sin(t) * 0.5,
      class: 0,
    });
    
    data.push({
      x: 0.5 + r * Math.cos(t + Math.PI) * 0.5,
      y: 0.5 + r * Math.sin(t + Math.PI) * 0.5,
      class: 1,
    });
  }
  
  return data;
}

// Generate a circle dataset
export function generateCircle(): TrainingPoint[] {
  const points = 100;
  const data: TrainingPoint[] = [];
  
  for (let i = 0; i < points; i++) {
    const r = Math.random() * 0.2 + 0.05;
    const t = Math.random() * 2 * Math.PI;
    data.push({
      x: 0.5 + r * Math.cos(t),
      y: 0.5 + r * Math.sin(t),
      class: 0,
    });

    const r2 = Math.random() * 0.2 + 0.3;
    const t2 = Math.random() * 2 * Math.PI;
    data.push({
      x: 0.5 + r2 * Math.cos(t2),
      y: 0.5 + r2 * Math.sin(t2),
      class: 1,
    });
  }
  
  return data;
}

// Convert training data to GPU format
export function prepareTrainingData(trainingData: TrainingPoint[]) {
  const inputs = new Float32Array(trainingData.length * 2);
  const targets = new Float32Array(trainingData.length);

  trainingData.forEach((point, i) => {
    inputs[i * 2] = point.x;
    inputs[i * 2 + 1] = point.y;
    targets[i] = point.class;
  });

  return { inputs, targets };
}
