/** Safe DOM rendering helpers for dashboard cards and accessible tables. */
import { temperatureColor } from "./charts.js";

export const fmt = (value, digits = 1, unit = "") => value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)}${unit}` : "—";
export const clock = seconds => Number.isFinite(Number(seconds)) ? `${String(Math.floor(seconds/3600)).padStart(2,"0")}:${String(Math.floor(seconds/60)%60).padStart(2,"0")}` : "—";
function cell(row, text) { const td=document.createElement("td"); td.textContent=text; row.append(td); }

export function renderRacks(container, tbody, snapshot, config, onSelect) {
  container.replaceChildren(); tbody.replaceChildren();
  const zones = Array.from({length: config.topology.zones}, (_,id) => { const section=document.createElement("section"); section.className="zone"; const title=document.createElement("h3"); title.textContent=`Zone ${id+1}`; const racks=document.createElement("div"); racks.className="zone-racks"; section.append(title,racks); container.append(section); return racks; });
  snapshot.estimated_temperatures_c.forEach((temperature,index)=>{
    const zone=config.topology.rack_zone[index], state=temperature>=config.thresholds.hard_limit_c?"HARD LIMIT":temperature>=config.thresholds.hotspot_c?"HOTSPOT":"NORMAL";
    const button=document.createElement("button"); button.className=`rack ${state==="HOTSPOT"?"hot":state==="HARD LIMIT"?"hard":""}`; button.style.setProperty("--rack-color",temperatureColor(temperature,config.thresholds.hotspot_c,config.thresholds.hard_limit_c)); button.setAttribute("aria-label",`Rack ${index+1}, ${temperature.toFixed(1)} degrees Celsius, ${state}`);
    const id=document.createElement("span"); id.textContent=`RACK ${String(index+1).padStart(2,"0")}`; const flag=document.createElement("b"); flag.textContent=state==="NORMAL"?"":state; id.append(flag); const strong=document.createElement("strong"); strong.textContent=`${temperature.toFixed(1)} °C`; const util=document.createElement("span"); util.textContent=`${(snapshot.utilization[index]*100).toFixed(0)}% utilized`;
    button.append(id,strong,util); button.onclick=()=>onSelect(index); zones[zone].append(button);
    const tr=document.createElement("tr"); cell(tr,`Rack ${index+1}`); cell(tr,`Zone ${zone+1}`); cell(tr,fmt(temperature,2," °C")); cell(tr,fmt(snapshot.inlet_temperatures_c[index],2," °C")); cell(tr,fmt(snapshot.utilization[index]*100,1,"%")); cell(tr,snapshot.sensor_valid[index]?"Valid":"Missing"); cell(tr,state); tbody.append(tr);
  });
}

export function renderCandidates(tbody, decision) {
  tbody.replaceChildren();
  if (!decision?.candidates?.length) { const row=document.createElement("tr"), td=document.createElement("td"); td.colSpan=7; td.className="empty-cell"; td.textContent="This controller did not perform a candidate search."; row.append(td); tbody.append(row); return; }
  decision.candidates.forEach(item=>{ const row=document.createElement("tr"),terms=item.objective_terms; const breakdown=terms?['energy','hotspot','hard_limit','imbalance','movement','migration'].map(key=>fmt(terms[key],2)).join(' / '):'—'; cell(row,item.label||item.action_id); cell(row,item.evaluated?"Yes":"No"); cell(row,fmt(item.predicted_peak_c,2," °C")); cell(row,fmt(item.predicted_energy_kwh,3," kWh")); cell(row,fmt(item.objective,3)); cell(row,breakdown); cell(row,item.evaluated?(item.predicted_feasible?"Feasible":"Predicted unsafe"):(item.rejection_reason||"Rejected")); tbody.append(row); });
}
