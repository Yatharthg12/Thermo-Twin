/** Small local Canvas chart layer; every chart also has an adjacent data table. */
const palette = { grid: "#29435b", text: "#91a7bd", cyan: "#61d9cf", blue: "#61a9ff", amber: "#f3b95f", red: "#ff7a83" };

function setup(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.getBoundingClientRect().width || canvas.clientWidth || 700));
  // Never read the mutable backing-store height here. On high-DPI displays the
  // previous implementation multiplied that value on every redraw, causing an
  // unbounded canvas and page-height feedback loop.
  const height = Number(canvas.dataset.chartHeight) || 300;
  const backingWidth = Math.max(1, Math.round(width * ratio));
  const backingHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

export function lineChart(canvas, series, options = {}) {
  const all = series.flatMap(s => s.values).filter(v => Number.isFinite(v.y));
  if (!all.length) { canvas.classList.add("is-empty"); return; }
  canvas.classList.remove("is-empty");
  const { ctx, width, height } = setup(canvas), margin = { left: 46, right: 18, top: 28, bottom: 34 };
  const minX = Math.min(...all.map(v => v.x)), maxX = Math.max(...all.map(v => v.x));
  let minY = options.minY ?? Math.floor(Math.min(...all.map(v => v.y)) - 1), maxY = options.maxY ?? Math.ceil(Math.max(...all.map(v => v.y)) + 1);
  if (maxY <= minY) maxY = minY + 1;
  const x = value => margin.left + (value - minX) / Math.max(maxX - minX, 1) * (width - margin.left - margin.right);
  const y = value => height - margin.bottom - (value - minY) / (maxY - minY) * (height - margin.top - margin.bottom);
  ctx.clearRect(0, 0, width, height); ctx.font = "11px system-ui"; ctx.fillStyle = palette.text; ctx.strokeStyle = palette.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const value = minY + (maxY-minY)*i/4, py = y(value); ctx.beginPath(); ctx.moveTo(margin.left,py); ctx.lineTo(width-margin.right,py); ctx.stroke(); ctx.fillText(value.toFixed(1),5,py+4); }
  if (options.thresholds) options.thresholds.forEach(item => { ctx.save(); ctx.strokeStyle=item.color; ctx.setLineDash([5,5]); ctx.beginPath(); ctx.moveTo(margin.left,y(item.value)); ctx.lineTo(width-margin.right,y(item.value)); ctx.stroke(); ctx.restore(); });
  series.forEach((item,index) => { ctx.strokeStyle=item.color || [palette.cyan,palette.blue,palette.amber,palette.red][index%4]; ctx.lineWidth=item.width||2; ctx.setLineDash(item.dash||[]); ctx.beginPath(); item.values.forEach((point,i) => { if (!Number.isFinite(point.y)) return; const px=x(point.x),py=y(point.y); i ? ctx.lineTo(px,py) : ctx.moveTo(px,py); }); ctx.stroke(); ctx.setLineDash([]); });
  let legendX=margin.left; series.forEach((item,index)=>{ const color=item.color||[palette.cyan,palette.blue,palette.amber,palette.red][index%4]; ctx.fillStyle=color; ctx.fillRect(legendX,8,13,3); ctx.fillStyle=palette.text; ctx.fillText(item.name,legendX+18,12); legendX += ctx.measureText(item.name).width+48; });
  ctx.fillStyle=palette.text; ctx.fillText(options.xLabel || "simulated minutes", Math.max(margin.left,(width-100)/2), height-6);
}

export function temperatureColor(value, hot = 35, hard = 40) {
  if (value >= hard) return "#ff6673";
  if (value >= hot) return "#f3b95f";
  const t = Math.max(0, Math.min(1, (value - 22) / (hot - 22)));
  return t > .55 ? "#57c7a5" : "#4a9ee9";
}
