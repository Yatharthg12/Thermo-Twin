/** ThermoTwin dashboard state, controls, polling, and real backend rendering. */
import { api, post } from "./api.js";
import { lineChart } from "./charts.js";
import { clock, fmt, renderCandidates, renderRacks } from "./components.js";

const $ = selector => document.querySelector(selector);
const state = { config:null, simulationId:null, simulation:null, selectedRack:0, jobId:null, jobActive:false, polling:false, jobPollTimer:null };
const titles = {overview:"Overview",thermal:"Thermal map",forecasts:"Forecasts",decisions:"Controller decisions",experiments:"Experiments",runs:"Run history"};

function message(text, error=false) { const box=$("#global-message"); if(!text){box.hidden=true;return;} box.hidden=false; box.classList.toggle("error",error); box.textContent=text; }
function setStatus(value) { const el=$("#session-status"); el.dataset.state=value; el.querySelector("span").textContent=value[0].toUpperCase()+value.slice(1); }
function setControls(enabled,status="") { ["#play","#pause","#step","#reset","#refresh-forecast"].forEach(id=>$(id).disabled=!enabled); if(enabled){ $("#play").disabled=status==="running"; $("#pause").disabled=status!=="running"; $("#step").disabled=status==="running"; } }

async function initialize() {
  try {
    const [health,config] = await Promise.all([api("/api/health"),api("/api/config")]); state.config=config;
    $("#model-badge").textContent=health.models_available?"ML artifacts ready":"Simulation-only mode"; $("#model-badge").className=`badge ${health.models_available?"ready":"warning"}`;
    config.scenarios.forEach(name=>$("#scenario").add(new Option(name.replaceAll("_"," "),name)));
    config.controllers.forEach(name=>{ const option=new Option(name.replaceAll("_"," "),name); if(name==="hybrid_mpc"&&!config.models_available){option.disabled=true;option.title="Run smoke preparation first";} $("#controller").add(option); }); $("#controller").value=config.models_available?"hybrid_mpc":"reactive";
    for(let rack=0;rack<config.topology.racks;rack++){ $("#forecast-rack").add(new Option(`Rack ${rack+1}`,rack)); }
    $("#research-workload").textContent=config.research_workload; setStatus("initialized"); await loadRuns(); await restoreJob();
  } catch(error) { message(`Initialization failed: ${error.message}`,true); setStatus("failed"); }
}

document.querySelectorAll("[role=tab]").forEach(tab=>tab.addEventListener("click",()=>{
  document.querySelectorAll("[role=tab]").forEach(item=>item.setAttribute("aria-selected",String(item===tab)));
  document.querySelectorAll(".panel").forEach(panel=>{ const active=panel.id===`panel-${tab.dataset.tab}`; panel.hidden=!active; panel.classList.toggle("active",active); });
  $("#page-title").textContent=titles[tab.dataset.tab]; if(["runs","experiments"].includes(tab.dataset.tab)) loadRuns(); if(tab.dataset.tab==="forecasts"&&state.simulationId) loadForecast(); if(tab.dataset.tab==="decisions"&&state.simulationId) loadDecisions();
}));

$("#new-simulation").addEventListener("click",async()=>{ try{ message(""); setStatus("initializing"); const data=await post("/api/simulations",{scenario:$("#scenario").value,controller:$("#controller").value,speed:Number($("#speed").value)}); state.simulationId=data.simulation_id; renderSimulation(data); setControls(true,data.status); message(`Simulation ${data.simulation_id} initialized. Use play or single-step to advance simulated time.`); }catch(error){message(error.message,true);setStatus("failed");} });
async function command(command,extra={}){ if(!state.simulationId)return; try{ const data=await post(`/api/simulations/${state.simulationId}/control`,{command,...extra}); renderSimulation(data); }catch(error){message(error.message,true);} }
$("#play").onclick=()=>command("play"); $("#pause").onclick=()=>command("pause"); $("#step").onclick=()=>command("step"); $("#reset").onclick=()=>command("reset"); $("#speed").onchange=()=>state.simulationId&&command("speed",{speed:Number($("#speed").value)});

function renderSimulation(data){ state.simulation=data; setStatus(data.status); setControls(true,data.status); const s=data.snapshot,m=data.metrics,c=state.config; $("#kpi-clock").textContent=clock(s.time_s); const peak=Math.max(...s.estimated_temperatures_c); $("#kpi-peak").textContent=fmt(peak,1," °C"); $("#kpi-thermal-state").textContent=peak>=c.thresholds.hard_limit_c?"hard-limit proxy exceeded":peak>=c.thresholds.hotspot_c?"hotspot threshold exceeded":"within configured threshold"; const power=s.it_power_w.reduce((a,b)=>a+b,0)+s.cooling_power_w+s.fan_power_w; $("#kpi-power").textContent=fmt(power/1000,1," kW"); $("#kpi-energy").textContent=fmt(m.total_energy_kwh,2," kWh"); $("#kpi-pue").textContent=fmt(m.pue_cooling_only,2); $("#kpi-service").textContent=fmt(m.service_fraction==null?null:m.service_fraction*100,1,"%"); renderCooling(s); renderRacks($("#rack-map"),$("#rack-table"),s,c,rack=>{state.selectedRack=rack;$("#forecast-rack").value=rack;document.querySelector('[data-tab="forecasts"]').click();}); refreshTrace(); }
function renderCooling(s){ const node=$("#cooling-state"); node.replaceChildren(); s.supply_actual_c.forEach((value,zone)=>{ const row=document.createElement("div"); row.className="state-row"; const label=document.createElement("span"); label.textContent=`Zone ${zone+1}`; const bar=document.createElement("div");bar.className="bar";const fill=document.createElement("i");fill.style.width=`${Math.max(2,Math.min(100,s.airflow_actual[zone]/1.35*100))}%`;bar.append(fill);const val=document.createElement("strong");val.textContent=`${value.toFixed(1)} °C · ${s.airflow_actual[zone].toFixed(2)} flow`;row.append(label,bar,val);node.append(row); }); }
async function refreshTrace(){ if(state.polling||!state.simulationId)return; state.polling=true; try{ const data=await api(`/api/simulations/${state.simulationId}/history?limit=180`); const history=data.history; const series=["min","mean","max"].map((name,index)=>({name, color:["#61a9ff","#61d9cf","#f3b95f"][index],values:history.map(item=>{const t=item.estimated_temperatures_c;return{x:item.time_s/60,y:name==="min"?Math.min(...t):name==="max"?Math.max(...t):t.reduce((a,b)=>a+b,0)/t.length};})})); lineChart($("#overview-chart"),series,{thresholds:[{value:state.config.thresholds.hotspot_c,color:"#f3b95f"},{value:state.config.thresholds.hard_limit_c,color:"#ff7a83"}]}); }catch(error){message(error.message,true);}finally{state.polling=false;} }
setInterval(async()=>{ if(!state.simulationId)return; try{const data=await api(`/api/simulations/${state.simulationId}`);renderSimulation(data);}catch(error){message(error.message,true);}},1000);

$("#forecast-rack").onchange=event=>{state.selectedRack=Number(event.target.value);loadForecast();}; $("#forecast-horizon").onchange=()=>loadForecast(); $("#refresh-forecast").onclick=()=>loadForecast();
async function loadForecast(){ if(!state.simulationId)return; try{ const data=await api(`/api/simulations/${state.simulationId}/forecast?rack_id=${state.selectedRack}&horizon_min=${$("#forecast-horizon").value}`); $("#forecast-origin").textContent=`ORIGIN ${clock(data.forecast_origin_s)} · RACK ${data.rack_id+1}`; $("#forecast-model-state").textContent=data.model_status; $("#forecast-model-state").className=`badge ${data.interval_level?"ready":"warning"}`; $("#hotspot-probability").textContent=data.hotspot_probability==null?"Unavailable":`${(data.hotspot_probability*100).toFixed(1)}%`; const history=data.history.map(v=>({x:v.time_s/60,y:v.temperature_c})),physics=data.future.map(v=>({x:v.time_s/60,y:v.physics_c})),hybrid=data.future.map(v=>({x:v.time_s/60,y:v.hybrid_c})),lower=data.future.map(v=>({x:v.time_s/60,y:v.lower_c})),upper=data.future.map(v=>({x:v.time_s/60,y:v.upper_c})); lineChart($("#forecast-chart"),[{name:"history",values:history,color:"#91a7bd"},{name:"physics",values:physics,color:"#61a9ff",dash:[5,4]},{name:"hybrid",values:hybrid,color:"#61d9cf"},{name:"lower",values:lower,color:"#315571",dash:[2,4]},{name:"upper",values:upper,color:"#315571",dash:[2,4]}],{thresholds:[{value:state.config.thresholds.hotspot_c,color:"#f3b95f"},{value:state.config.thresholds.hard_limit_c,color:"#ff7a83"}]}); const tbody=$("#forecast-table");tbody.replaceChildren();data.future.forEach(v=>{const tr=document.createElement("tr");[clock(v.time_s),fmt(v.physics_c,2," °C"),fmt(v.hybrid_c,2," °C"),fmt(v.lower_c,2," °C"),fmt(v.upper_c,2," °C")].forEach(text=>{const td=document.createElement("td");td.textContent=text;tr.append(td);});tbody.append(tr);}); }catch(error){message(`Forecast unavailable: ${error.message}`,true);} }

async function loadDecisions(){ if(!state.simulationId)return; try{const data=await api(`/api/simulations/${state.simulationId}/decisions`),decision=data.decisions.at(-1);if(!decision){renderCandidates($("#candidate-table"),null);return;} const chosen=decision.chosen_action;$("#decision-action").textContent=chosen?.label||chosen?.action_id||"Fallback";$("#decision-explanation").textContent=decision.explanation;$("#decision-latency").textContent=fmt(decision.latency_ms,1," ms");$("#decision-feasibility").textContent=decision.fallback?"Fallback":"Preferred feasible";renderCandidates($("#candidate-table"),decision);}catch(error){message(error.message,true);} }
setInterval(()=>{const visible=!$("#panel-decisions").hidden;if(visible)loadDecisions();},1800);

const activeJobStates = new Set(["queued","running","cancelling"]);
function updateLaunchButtons(active=false) {
  $("#run-smoke").disabled=active;
  $("#run-research").disabled=active || !$("#research-confirm").checked;
}
function renderJob(job) {
  const active=activeJobStates.has(job.status);
  state.jobActive=active;
  $("#job-title").textContent=`${job.profile} · ${job.status} · ${job.progress}%`;
  $("#job-progress").style.width=`${job.progress}%`;
  $("#job-log").textContent=job.logs.length?job.logs.join("\n"):"Waiting for the bounded worker…";
  $("#cancel-job").disabled=!['queued','running'].includes(job.status);
  updateLaunchButtons(active);
  if(active&&job.profile==="research") $("#job-note").textContent="Full research uses 10 replications and can take hours. You may leave this page and return; status is persisted.";
  else if(active) $("#job-note").textContent="Smoke normally reuses matching artifacts; a source or configuration change requires a bounded rebuild.";
  else if(job.status==="complete") $("#job-note").textContent=`Completed successfully${job.run_id?` · evidence run ${job.run_id}`:""}.`;
  else if(job.status==="cancelled") $("#job-note").textContent="Cancelled cooperatively. Any completed partial evidence remains in run history.";
  else $("#job-note").textContent=job.error||`Last job status: ${job.status}.`;
}
$("#research-confirm").onchange=()=>updateLaunchButtons(state.jobActive);
document.querySelectorAll(".launch-job").forEach(button=>button.onclick=async()=>{
  try {
    const profile=button.dataset.profile;
    if(profile==="research"&&!$("#research-confirm").checked) throw new Error("Confirm the research compute warning before launch.");
    const job=await post("/api/jobs",{profile,confirm_research:profile==="research"});
    state.jobId=job.job_id; renderJob(job); message(`Background ${job.profile} job queued. Progress persists if you change tabs.`); pollJob();
  } catch(error){message(error.message,true);}
});
$("#cancel-job").onclick=async()=>{if(state.jobId)try{const job=await post(`/api/jobs/${state.jobId}/cancel`);renderJob(job);pollJob();}catch(error){message(error.message,true);}};
async function restoreJob(){
  try {
    const data=await api("/api/jobs");
    const job=data.jobs.find(item=>activeJobStates.has(item.status))||data.jobs[0];
    if(!job){updateLaunchButtons(false);return;}
    state.jobId=job.job_id; renderJob(job); if(activeJobStates.has(job.status))pollJob();
  } catch(error){message(`Job history unavailable: ${error.message}`,true);}
}
async function pollJob(){
  if(!state.jobId)return;
  clearTimeout(state.jobPollTimer);
  try {
    const job=await api(`/api/jobs/${state.jobId}`); renderJob(job);
    if(activeJobStates.has(job.status)) state.jobPollTimer=setTimeout(pollJob,1200);
    else {if(job.error)message(job.error,true);await loadRuns();}
  } catch(error){message(`Job status temporarily unavailable: ${error.message}`,true);state.jobPollTimer=setTimeout(pollJob,2500);}
}

async function loadRuns(){
  try {
    const data=await api("/api/runs"),container=$("#run-list"),results=$("#experiment-results");
    container.replaceChildren(); results.replaceChildren(); $("#benchmark-tables").replaceChildren();
    if(!data.runs.length){const card=document.createElement("article");card.className="card empty";card.textContent="No persisted experiment runs found.";container.append(card);return;}
    const evidenceRun=data.runs.find(run=>run.status==="complete")||data.runs[0];
    data.runs.forEach(run=>{
      const card=document.createElement("article"); card.className="card run-card";
      const header=document.createElement("div"); header.className="run-header";
      const info=document.createElement("div"),title=document.createElement("strong"),meta=document.createElement("div");
      title.textContent=run.run_id; meta.className="run-meta"; meta.textContent=`${run.profile} · ${run.status} · ${run.wall_time_s?run.wall_time_s.toFixed(1)+" s":"duration unavailable"}`; info.append(title,meta);
      const badge=document.createElement("span"); badge.className=`badge ${run.status==="complete"?"ready":"warning"}`; badge.textContent=run.status; header.append(info,badge);
      const links=document.createElement("div"); links.className="artifact-links";
      run.artifacts.filter(a=>["report.html","summary.json","experiment_2_forecasts.csv","experiment_4_control.csv","experiment_5_ablation.csv","experiment_6_robustness.csv"].includes(a.name)).forEach(a=>{const link=document.createElement("a");link.href=`/api/runs/${encodeURIComponent(run.run_id)}/artifacts/${a.artifact_id.split('/').map(encodeURIComponent).join('/')}`;link.textContent=a.name;link.target=a.name.endsWith(".html")?"_blank":"_self";links.append(link);});
      card.append(header,links); container.append(card);
      if(run.run_id===evidenceRun.run_id){[1,2,3,4,5,6].forEach(n=>{const result=document.createElement("article");result.className="card result-card";result.innerHTML=`<span>EXPERIMENT ${n}</span><strong>${["Physics & numerics","Forecast benchmark","Horizon sensitivity","Closed-loop control","Ablation","Robustness & scale"][n-1]}</strong><span>${run.status==="complete"?`Artifacts from ${run.run_id}`:"Run incomplete"}</span>`;results.append(result);});}
    });
    await loadExperimentTables(evidenceRun);
  } catch(error){message(`Run history unavailable: ${error.message}`,true);}
}

function parseCsv(text){ const lines=text.trim().split(/\r?\n/); const split=line=>line.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/).map(value=>value.replace(/^\"|\"$/g,"").replaceAll('""','"')); const headers=split(lines.shift()||""); return lines.map(line=>Object.fromEntries(split(line).map((value,index)=>[headers[index],value]))); }
async function loadExperimentTables(run){ const container=$("#benchmark-tables"); container.replaceChildren(); const specs=[
  ["Forecast benchmark · held-out test", "experiment_2_forecasts.csv", ["model","horizon_min","mae_c","rmse_c","count"]],
  ["Closed-loop comparison", "experiment_4_control.csv", ["scenario","controller","total_energy_kwh","peak_temperature_c","service_fraction"]],
  ["Component ablation", "experiment_5_ablation.csv", ["controller","total_energy_kwh","peak_temperature_c","service_fraction"]],
]; for(const [title,name,columns] of specs){const artifact=run.artifacts.find(item=>item.name===name);if(!artifact)continue;const response=await fetch(`/api/runs/${encodeURIComponent(run.run_id)}/artifacts/${artifact.artifact_id}`);if(!response.ok)continue;const rows=parseCsv(await response.text()).slice(0,name.includes("forecasts")?12:20);const card=document.createElement("article");card.className="card";const heading=document.createElement("h3");heading.textContent=title;const scroll=document.createElement("div");scroll.className="table-scroll";const table=document.createElement("table"),thead=document.createElement("thead"),tr=document.createElement("tr"),tbody=document.createElement("tbody");columns.forEach(column=>{const th=document.createElement("th");th.textContent=column.replaceAll("_"," ");tr.append(th);});thead.append(tr);rows.forEach(row=>{const line=document.createElement("tr");columns.forEach(column=>{const td=document.createElement("td");const numeric=Number(row[column]);td.textContent=Number.isFinite(numeric)&&row[column]!==""?numeric.toFixed(column.includes("energy")?3:column.includes("temperature")||column.includes("mae")||column.includes("rmse")?2:0):row[column];line.append(td);});tbody.append(line);});table.append(thead,tbody);scroll.append(table);card.append(heading,scroll);container.append(card);} }

window.addEventListener("resize",()=>state.simulationId&&refreshTrace()); initialize();
