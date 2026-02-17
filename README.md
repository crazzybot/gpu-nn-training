# WebGPU Neural Network Training

A high-performance, browser-based neural network training application that leverages **WebGPU** for GPU-accelerated computation. Train a 2-layer neural network entirely on your GPU and watch the decision boundary form in real-time.

## Features

- **GPU Acceleration**: Full forward and backward propagation implemented in WebGPU compute shaders
- **CPU Fallback**: Switch to JavaScript-based training for comparison or compatibility
- **Interactive Data Creation**: Click on the canvas to add training points (red for class 0, blue for class 1)
- **Dataset Generation**: Automatically generate spiral or circular datasets
- **Real-time Visualization**: Watch the decision boundary adapt as the model learns
- **Loss Tracking**: Monitor training progress with a live loss chart
- **Responsive Design**: Runs smoothly in your browser

## Tech Stack

- **WebGPU**: GPU compute for neural network training
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Canvas API**: Real-time visualization

## Project Structure

```
├── index.html              # Main HTML entry point
├── package.json            # Project dependencies
├── src/
│   ├── main.ts            # Application entry point and event handling
│   ├── webgpu.ts          # WebGPU initialization and utilities
│   ├── neural-network.ts  # Network architecture configuration
│   ├── gpu-training.ts    # GPU compute pipeline and training logic
│   ├── cpu-training.ts    # CPU-based training implementation
│   ├── gpu-buffers.ts     # GPU buffer management
│   ├── data-management.ts # Training data generation and preparation
│   ├── visualization.ts   # Canvas rendering and visualization
│   ├── ui-handlers.ts     # UI state and event handlers
│   ├── shaders.ts         # WebGPU shader code
│   └── style.css          # Application styles
└── vite.config.ts         # Build configuration
```

## Getting Started

### Prerequisites

- A modern browser that supports WebGPU:
  - Chrome/Edge 113+
  - Firefox 119+ (with user permission)
  - Safari 18+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpu-nn-training
```

2. Install dependencies:
```bash
pnpm install
# or npm install / yarn install
```

3. Start the development server:
```bash
pnpm dev
```

4. Open your browser to the local development URL (typically `http://localhost:5173`)

## How to Use

1. **Add Training Data**:
   - Click on the "Training Data" canvas to manually add points
   - Each click cycles between class 0 (red) and class 1 (blue)
   - Or use "Generate Spiral" or "Generate Circle" for automatic datasets

2. **Select Training Mode**:
   - Choose **GPU (WebGPU)** for fast GPU-accelerated training
   - Or **CPU (JavaScript)** for JavaScript-based training

3. **Train the Model**:
   - Click "Start Training" to begin
   - Watch the decision boundary form in real-time
   - Monitor the loss chart to track learning progress

4. **Clear and Restart**:
   - Use "Clear Data" to remove all data points
   - The model automatically resets for new training runs

## Network Architecture

The neural network consists of:
- **Input Layer**: 2 neurons (for 2D data points)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation

Binary cross-entropy loss is used for the classification task.

## Performance

The GPU implementation significantly outperforms CPU training:
- **GPU**: Process thousands of training iterations per second
- **CPU**: Slower but useful for debugging and comparison

You can switch between modes to see the performance difference yourself.

## Building for Production

```bash
pnpm build
```

This generates optimized static files in the `dist` directory, which can be deployed to any static hosting service.

## Browser Preview

After building, you can preview the production build locally:
```bash
pnpm preview
```

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome  | 113+    | ✅ Full support |
| Edge    | 113+    | ✅ Full support |
| Firefox | 119+    | ⚠️ Behind flag |
| Safari  | 18+     | ⚠️ Limited support |

## License

ISC

## Author

Crazzybot
