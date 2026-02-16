// Update status message in the UI
export function updateStatus(
  message: string,
  type: "error" | "success" | "info"
) {
  const statusDiv = document.getElementById("status");
  if (statusDiv) {
    statusDiv.textContent = message;
    statusDiv.className =
      type === "error"
        ? "status-error"
        : type === "success"
          ? "status-success"
          : "status-info";
  }
}

// Update training metrics in the UI
export function updateMetrics(metrics: {
  epoch: number;
  loss: number;
  accuracy: number;
  time: number;
}) {
  const epochEl = document.getElementById("epoch");
  const lossEl = document.getElementById("loss");
  const accuracyEl = document.getElementById("accuracy");
  const timeEl = document.getElementById("time");

  if (epochEl) epochEl.textContent = metrics.epoch.toString();
  if (lossEl) lossEl.textContent = metrics.loss.toFixed(4);
  if (accuracyEl) accuracyEl.textContent = metrics.accuracy.toFixed(1) + "%";
  if (timeEl) timeEl.textContent = metrics.time.toFixed(2) + "ms";
}

// Clear all metrics in the UI
export function clearMetrics() {
  const epochEl = document.getElementById("epoch");
  const lossEl = document.getElementById("loss");
  const accuracyEl = document.getElementById("accuracy");
  const timeEl = document.getElementById("time");

  if (epochEl) epochEl.textContent = "0";
  if (lossEl) lossEl.textContent = "-";
  if (accuracyEl) accuracyEl.textContent = "-";
  if (timeEl) timeEl.textContent = "-";
}

// Set button states
export function setButtonStates(states: {
  trainBtn?: boolean;
  stopBtn?: boolean;
}) {
  const trainBtn = document.getElementById("trainBtn") as HTMLButtonElement;
  const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;

  if (trainBtn && states.trainBtn !== undefined) trainBtn.disabled = states.trainBtn;
  if (stopBtn && states.stopBtn !== undefined) stopBtn.disabled = states.stopBtn;
}
