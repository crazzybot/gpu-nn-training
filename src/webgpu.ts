// Initialize WebGPU
async function initWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No GPU adapter found");
  }
  const requiredFeatures: GPUFeatureName[] = [];
  const requiredLimits = {
    maxStorageBuffersPerShaderStage: 10,
  };
  const device = await adapter.requestDevice({
    defaultQueue: {
      label: "my_queue",
    },
    requiredFeatures,
    requiredLimits,
  });
  return device;
}

export {initWebGPU};