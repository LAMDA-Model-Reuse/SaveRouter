const fields = {
  upfront: document.querySelector("#upfront"),
  baseline: document.querySelector("#baseline"),
  routed: document.querySelector("#routed"),
  overhead: document.querySelector("#overhead"),
  horizon: document.querySelector("#horizon"),
};

const canvas = document.querySelector("#payback-chart");
const context = canvas.getContext("2d");
const bepValue = document.querySelector("#bep-value");
const ratioValue = document.querySelector("#ratio-value");
const ratioCaption = document.querySelector("#ratio-caption");
const calculatorNote = document.querySelector("#calculator-note");
const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

function readValues() {
  return Object.fromEntries(Object.entries(fields).map(([name, input]) => [name, Number(input.value)]));
}

function pointAt(x, maxX, maxY, width, height, padding, cost) {
  return {
    x: padding.left + (x / maxX) * (width - padding.left - padding.right),
    y: height - padding.bottom - (cost / maxY) * (height - padding.top - padding.bottom),
  };
}

function drawLine(points, color) {
  context.beginPath();
  points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
  context.strokeStyle = color;
  context.lineWidth = 4;
  context.lineCap = "round";
  context.stroke();
}

function drawChart(values, breakEven) {
  const { upfront, baseline, routed, overhead, horizon } = values;
  const width = canvas.width;
  const height = canvas.height;
  const padding = { top: 35, right: 35, bottom: 50, left: 72 };
  const endBaseline = baseline * horizon;
  const endRouter = upfront + (routed + overhead) * horizon;
  const maxY = Math.max(endBaseline, endRouter, 1) * 1.08;
  context.clearRect(0, 0, width, height);

  context.font = "22px Inter, sans-serif";
  context.fillStyle = "rgba(255,255,255,.42)";
  context.strokeStyle = "rgba(255,255,255,.08)";
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const x = padding.left + (index / 4) * (width - padding.left - padding.right);
    const y = padding.top + (index / 4) * (height - padding.top - padding.bottom);
    context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
    context.textAlign = "center";
    context.fillText(numberFormat.format(horizon * index / 4), x, height - 15);
  }
  context.save();
  context.translate(18, height / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("Cumulative cost", 0, 0);
  context.restore();

  const xValues = Array.from({ length: 80 }, (_, index) => horizon * index / 79);
  const baselinePoints = xValues.map(x => pointAt(x, horizon, maxY, width, height, padding, baseline * x));
  const routerPoints = xValues.map(x => pointAt(x, horizon, maxY, width, height, padding, upfront + (routed + overhead) * x));
  drawLine(baselinePoints, "#8b7bf6");
  drawLine(routerPoints, "#f4b860");

  if (Number.isFinite(breakEven) && breakEven <= horizon) {
    const point = pointAt(breakEven, horizon, maxY, width, height, padding, baseline * breakEven);
    context.setLineDash([8, 8]);
    context.strokeStyle = "rgba(255,255,255,.45)";
    context.beginPath(); context.moveTo(point.x, point.y); context.lineTo(point.x, height - padding.bottom); context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#fff";
    context.beginPath(); context.arc(point.x, point.y, 6, 0, Math.PI * 2); context.fill();
    context.textAlign = point.x > width * .72 ? "right" : "left";
    context.fillStyle = "rgba(255,255,255,.72)";
    context.fillText(`Break-even: ${numberFormat.format(breakEven)}`, point.x + (point.x > width * .72 ? -12 : 12), point.y - 14);
  }
}

function updateCalculator() {
  const values = readValues();
  const valid = values.upfront >= 0 && values.baseline > 0 && values.routed >= 0 && values.overhead >= 0 && values.horizon > 0;
  const savings = values.baseline - values.routed - values.overhead;
  if (!valid) {
    calculatorNote.textContent = "Enter non-negative costs, a positive baseline cost, and a positive horizon.";
    calculatorNote.classList.add("error");
    bepValue.textContent = "—";
    ratioValue.textContent = "—";
    return;
  }
  calculatorNote.textContent = "Illustrative calculator; paper results use benchmark-specific costs.";
  calculatorNote.classList.remove("error");
  const breakEven = savings > 0 ? Math.ceil(values.upfront / savings) : Infinity;
  const ratio = (values.upfront + values.horizon * (values.routed + values.overhead)) / (values.horizon * values.baseline);
  bepValue.textContent = Number.isFinite(breakEven) ? numberFormat.format(breakEven) : "∞";
  ratioValue.textContent = ratio.toFixed(4);
  ratioCaption.textContent = ratio < 1 ? "paid back by horizon" : "not paid back by horizon";
  drawChart(values, breakEven);
}

Object.values(fields).forEach(input => input.addEventListener("input", updateCalculator));
window.addEventListener("resize", updateCalculator);
updateCalculator();

document.querySelector("#copy-citation").addEventListener("click", async event => {
  const text = document.querySelector("#citation").innerText;
  await navigator.clipboard.writeText(text);
  const button = event.currentTarget;
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = "Copy BibTeX"; }, 1600);
});
