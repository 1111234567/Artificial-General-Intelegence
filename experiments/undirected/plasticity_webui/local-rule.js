import {renderChart} from "./local-chart.js";

const arms = {control: "原规则", local: "逐步局部规则", modulated: "接收节点调制"};
const names = Object.keys(arms);
const $ = id => document.getElementById(id);
const set = (id, value) => { $(id).textContent = value; };
const finite = Number.isFinite;
const nf = new Intl.NumberFormat("zh-CN", {maximumFractionDigits:2});
const num = n => finite(n) ? nf.format(n) : "—";
const pct = n => finite(n) ? (100*n).toFixed(2)+"%" : "—";
const sc = n => finite(n) ? n.toFixed(4) : "—";
const pp = n => finite(n) ? (n>0?"+":"")+(100*n).toFixed(2) : "—";
const format = (n, unit) => unit === "percent" ? pct(n) : unit === "score" ? sc(n) : num(n);
const time = x => x && !Number.isNaN(Date.parse(x)) ? new Date(x).toLocaleString("zh-CN", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}) : "—";
const ratio = (a,b) => finite(a)&&b>0 ? Math.min(1,Math.max(0,a/b)) : 0;
const duration = n => { if(!finite(n)||n<0)return "采样中";const m=Math.ceil(n);return m>=60?Math.floor(m/60)+" 小时 "+m%60+" 分钟":m+" 分钟"; };
function readSetting(key,fallback) {try{return localStorage.getItem(key)??fallback;}catch{return fallback;}}
function saveSetting(key,value) {try{localStorage.setItem(key,value);}catch{}}

const fields = {
  strict:["严格完成率","percent"], recovery_rate:["补救完成率","percent"], completion:["总完成率","percent"],
  selection_score:["选拔得分","score"], first_finish_correct:["全程首次回答正确率","percent"],
  symbol_late_mean:["后半生 A、B 首答等权平均","percent"], symbol_late_a:["情境 A 后半生首答正确率","percent"],
  symbol_late_b:["情境 B 后半生首答正确率","percent"], pair_strict_rate:["双回访成功率","percent"],
  premature_request_rate:["抢答请求率","percent"], ready_coverage:["窗口内作答率","percent"],
  idle:["空闲端口动作率","percent"], post_answer_actions:["答对后多余动作次数","number"],
  initial_edges:["出生时活跃边","number"], final_edges:["最终活跃边","number"], mean_edges:["时均活跃边","number"],
  births:["新增边次数","number"], deaths:["退出边次数","number"], weight_updates:["权重改动次数","number"],
  bias_updates:["偏置改动次数","number"], weight_change:["权重累计改变量","number"], bias_change:["偏置累计改变量","number"]
};
const categories = {
  completion:{keys:["strict","recovery_rate","completion"], set:["strict","recovery_rate","completion"],multi:"严格、补救与总完成"},
  score:{keys:["selection_score"]},
  symbols:{keys:["pair_strict_rate","first_finish_correct","symbol_late_mean","symbol_late_a","symbol_late_b"],set:["symbol_late_mean","symbol_late_a","symbol_late_b"],multi:"后半生 A / B"},
  timing:{keys:["ready_coverage","premature_request_rate","idle","post_answer_actions"],set:["ready_coverage","premature_request_rate","idle"],multi:"窗口作答、抢答与空闲动作"},
  graph:{keys:["final_edges","initial_edges","mean_edges","births","deaths","weight_updates","bias_updates","weight_change","bias_change"],set:["initial_edges","mean_edges","final_edges"],multi:"出生、时均与最终活跃边"}
};
const palette=["#d4f983","#72e4b3","#83bdf6"];
const phaseName={search:"候选搜索",holdout:"留出评估",receipt_audit:"回执干预",final_comparison:"终代复评",complete:"已完成",failed:"运行失败",paused:"已暂停"};
let data={}, history=[], latest={}, category="completion", metricChoices={}, stream="stdout", logData={}, logProblem="";
let fit=readSetting("agi-plasticity-show-fit","false")==="true";
function options(select, list, preferred) {
  const key=JSON.stringify(list);
  if(select.dataset.options!==key) {
    const old=preferred??select.value;
    select.replaceChildren(...list.map(([value,label])=>new Option(label,value)));
    select.dataset.options=key;
    if(list.some(([v])=>v===old))select.value=old;
  } else if(preferred!==undefined && list.some(([v])=>v===preferred))select.value=preferred;
}
function generations(id, values) {
  const select=$(id);
  const sorted=[...new Set(values.filter(Number.isInteger))].sort((a,b)=>b-a);
  options(select,[["latest",sorted.length?"最新 · 第 "+sorted[0]+" 代":"等待评估"],...sorted.map(g=>[String(g),"第 "+g+" 代"])]);
  select.disabled=!sorted.length;
  return select.value==="latest"?sorted[0]:Number(select.value);
}
function notice(text) {set("notice",text);$("notice").hidden=!text;}
function values(id,keys,metrics,graph=false) {
  const container=$(id);
  if(!container.hasChildNodes()) {
    for(const key of keys){
      const group=document.createElement("div"), label=document.createElement("dt"), v=document.createElement("dd");
      label.textContent=fields[key][0];v.className="mono";v.dataset.key=key;
      if(["strict","pair_strict_rate"].includes(key))v.classList.add("private-primary");
      if(["births","deaths"].includes(key))group.className=key;
      group.append(label,v);container.append(group);
    }
  }
  for(const v of container.querySelectorAll("[data-key]")){
    const key=v.dataset.key; let text=format(metrics[key],fields[key][1]);
    v.textContent=text;
    if(graph)v.title=fields[key][0]+" · "+text;
    if(key==="pair_strict_rate"&&finite(metrics.eligible_pairs))v.title=num(metrics.pairs_both_strict)+" / "+num(metrics.eligible_pairs)+" 对";
  }
}
function renderTable(id,rows) {
  const el=$(id);const sig=JSON.stringify(rows);if(el.dataset.signature===sig)return;
  el.dataset.signature=sig;
  el.replaceChildren(...rows.map(row=>{
    const tr=document.createElement("tr");
    row.forEach((value,i)=>{
      const cell=document.createElement(i===0?"th":"td");
      if(i===0)cell.scope="row";
      if(Array.isArray(value)){cell.textContent=value[0];const hint=document.createElement("div");hint.className="ci-range";hint.textContent=value[1];cell.append(hint);}
      else cell.textContent=value;
      tr.append(cell);
    });return tr;
  }));
}
function updateProgress() {
  const p=data.progress||{}, proc=data.process||{}, phase=phaseName[p.phase]||p.phase||"准备中";
  set("run-name",data.run_name||"正在连接本地运行目录");
  let state=p.state||"waiting", title=state==="running"?phase+"中":({complete:"已完成",paused:"已暂停",failed:"运行失败",waiting:"等待训练"})[state]||state;
  if(state==="running"&&data.process_active===false){state="failed";title="进程已退出";}
  $("status").dataset.tone=state;set("status-text",title);
  const device=String(p.device||proc.device||"").toUpperCase();
  set("run-detail",[device&&("设备 "+device),device==="CUDA"&&"FP16 循环计算 / FP32 累加",proc.pid&&("PID "+proc.pid)].filter(Boolean).join(" · "));
  const current=arms[p.current_arm];
  set("phase-detail",p.phase==="search"?(current?current+" 正在评估候选":"正在读取候选批次"):
    p.phase==="holdout"?((current||"")+" · 已评估 "+num(p.audit_seeds_completed)+" / "+num(p.audit_seeds_total)+" 个留出种子"):
    phase+(current?" · "+current:""));
  set("overall-percent",finite(p.total_candidates)&&p.total_candidates>0?pct(ratio(p.completed_candidates,p.total_candidates)):"—");
  set("overall-count",num(p.completed_candidates)+" / "+num(p.total_candidates));
  $("overall-bar").value=ratio(p.completed_candidates,p.total_candidates);
  set("group-progress-label",(current?current+" · ":"")+"本代候选");
  set("generation-count",num(p.generation_completed)+" / "+num(p.generation_total));
  $("generation-bar").value=ratio(p.generation_completed,p.generation_total);
  set("generation",num(p.generation)+" / "+num(p.generations));
  set("rate",num(p.candidates_per_minute)+" / 分钟");
  set("remaining",p.state==="complete"?"已完成":p.state==="running"&&data.process_active!==false?duration((p.total_candidates-p.completed_candidates)/p.candidates_per_minute):"—");
  set("updated",time(p.updated_utc));set("started",time(p.started_utc||proc.started_utc));
  const errors=Object.keys(data.read_errors||{});const age=Date.now()-Date.parse(p.updated_utc);
  notice(p.failure|| (errors.length?"暂时无法读取 "+errors.join("、"):
    p.state==="running"&&phaseName[p.phase]===phaseName.search&&age>120000?"最近两分钟没有新的训练记录，当前显示最近一次写入。":""));
}
function readout() {
  const p=data.progress||{}, selected=$("inspect-arm").value, arm=selected==="follow"?(p.current_arm||"control"):selected;
  const running=p.latest?.arm===arm && p.latest.generation===p.generation ? p.latest:null;
  const saved=p.arms_progress?.[arm]?.latest||latest[arm];
  const entry=running||saved;
  const c=entry?.champion||{}, point=entry?"第 "+entry.generation+" 代":null;
  const label=entry?.sealed===false?"已评 "+num(entry.candidates)+" / "+num(p.population):"本代已完成";
  set("candidate-title",(arms[arm]||"")+" · 领先候选");
  set("candidate-id",entry?[point,label,c.id].filter(Boolean).join(" · "):"等待首批候选");
  set("graph-source",entry?arms[arm]+" · "+point+"领先候选。边数按生命平均，改动次数按 8 个生命累计。":"等待图结构记录");
  values("completion-cells",["strict","recovery_rate","completion","selection_score"],c);
  values("timing-cells",["ready_coverage","premature_request_rate","idle","post_answer_actions"],c);
  values("symbol-cells",["symbol_late_a","symbol_late_b","first_finish_correct","pair_strict_rate"],c);
  values("graph-cells",["initial_edges","final_edges","mean_edges","weight_updates","births","deaths","bias_updates","weight_change","bias_change"],c,true);
}
function compare() {
  const counts=new Map();
  for(const row of history){
    if(!counts.has(row.generation))counts.set(row.generation,new Set());
    counts.get(row.generation).add(row.arm);
  }
  const g=generations("compare-gen",Array.from(counts).filter(([,a])=>a.size===names.length).map(([g])=>g));
  set("compare-detail",g?"第 "+g+" 代 · 三组已全部评估，比较各组当代选拔冠军。":"等待三组完成同一代。");
  renderTable("comparison-rows",g?names.map(arm=>{
    const c=history.find(row=>row.arm===arm&&row.generation===g)?.champion||{};
    return [arms[arm],pct(c.strict),pct(c.recovery_rate),[pct(c.pair_strict_rate),num(c.pairs_both_strict)+" / "+num(c.eligible_pairs)+" 对"],sc(c.selection_score)];
  }):[]);
  set("checkpoint",names.map(a=>arms[a]+" "+(latest[a]?.generation??0)+" 代").join(" · "));
}
function audits(){
  const audits=data.holdout_history?.length?data.holdout_history:(data.holdout?.generation?[data.holdout]:[]);
  const generation=generations("holdout-gen",audits.map(a=>a.generation));
  const actual=audits.find(a=>a.generation===generation);
  set("holdout-detail",actual?"第 "+generation+" 代 · 四岛冠军"+(actual.seed_count?"在 "+actual.seed_count+" 个未见世界上的平均成绩。":"的未见世界平均成绩。"):"等待首次留出评估。");
  renderTable("holdout-rows",actual?names.map(arm=>{
    const a=actual.arms?.[arm]||{};return [arms[arm],pct(a.strict),pct(a.completion),pct(a.first_finish_correct),pct(a.pair_strict_rate),sc(a.selection_score)];
  }):[]);
  const versions=data.receipt_history||[];
  const version=generations("receipt-gen",versions.map(v=>v.generation));
  const receipts=versions.find(v=>v.generation===version)?.arms||{};
  const lag=$("receipt-delay").value;
  set("receipt-detail",version?"第 "+version+" 代 · "+(lag==="0"?"立即回访":"隔 "+lag+" 道干扰题回访")+"。真实回执相对颠倒回执的首答正确率差，单位为百分点。括号内为按世界配对的 95% 区间。":"等待回执干预评估。");
  renderTable("receipt-rows",version?names.map(arm=>{
    const a=receipts[arm], d=a?.delays?.[lag];
    const effect=key=>{const v=d?.[key+"_truthful_correct_gain"];return v&&finite(v.mean)?[pp(v.mean),v.ci95?"("+v.ci95.map(pp).join(", ")+")":""]: "—";};
    return [arms[arm],a?num(a.events)+" / "+num(a.opportunities):"等待评估",effect("same"),effect("other_context"),effect("other_raw")];
  }):[]);
}
function renderTrend(){
  const cat=categories[category], arm=$("chart-arm").value, select=$("chart-metric");
  const list=cat.keys.map(k=>[k,fields[k][0]]);
  if(arm!=="all"&&cat.set)list.unshift(["set",cat.multi]);
  const pref=metricChoices[category+"-"+arm];
  options(select,list,pref&&list.some(([k])=>k===pref)?pref:(arm!=="all"&&cat.set?"set":cat.keys[0]));
  const key=select.value, combined=key==="set";
  const allowBoth=arm!=="all"&&!combined;
  const scopeSelect=$("chart-scope");
  const bothOption=scopeSelect.querySelector('option[value="both"]');
  bothOption.disabled=!allowBoth;
  if(!allowBoth&&scopeSelect.value==="both")scopeSelect.value="champion";
  const scope=scopeSelect.value, base=combined?cat.set[0]:key;
  const metric={key:base,label:combined?cat.multi:fields[base][0], unit:fields[base][1]};
  if(arm!=="all") {
    if(combined)metric.series=cat.set.map((k,i)=>({arm,key:k,label:fields[k][0],color:palette[i]}));
    else if(scope==="both")metric.series=["champion","mean"].map((s,i)=>({arm,key,label:s==="mean"?"全代均值":"本代冠军",scope:s,color:palette[i]}));
    else metric.series=[{arm,key,label:arms[arm],color:palette[names.indexOf(arm)]}];
  }
  set("trend-detail",(arm==="all"?"三组对比":arms[arm])+" · "+metric.label+" · 已完成代次"+(fit?"。亮线为局部拟合，淡线为实测。":""));
  for(const b of $("chart-tabs").querySelectorAll("[data-chart]"))b.setAttribute("aria-pressed",String(b.dataset.chart===category));
  $("fit").setAttribute("aria-pressed",String(fit));
  renderChart({canvas:$("chart"),tooltip:$("chart-tooltip"),legend:$("chart-legend"),history,metric,scope:scope==="both"?"champion":scope,fit});
}
function renderLog(){
  const pre=$("log-text"), record=logData[stream];
  set("log-detail",(stream==="stdout"?"标准输出":"标准错误")+(record?.truncated?" · 最近日志":""));
  document.querySelectorAll("[data-stream]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.stream===stream)));
  const text=logProblem||(record?.error?"暂时无法读取日志":record?.text||(record?(stream==="stderr"?"暂无标准错误输出":"等待标准输出"):"正在读取日志"));
  if(pre.textContent!==text){
    const follow=pre.scrollHeight-pre.scrollTop-pre.clientHeight<8;pre.textContent=text;
    if(follow)pre.scrollTop=pre.scrollHeight;
  }
}
function render(snapshot){
  data=snapshot;history=data.history||[];latest={};
  for(const row of history)if(!latest[row.arm]||latest[row.arm].generation<row.generation)latest[row.arm]=row;
  updateProgress();readout();compare();audits();renderTrend();
}
async function get(path){const response=await fetch(path,{cache:"no-store"});if(!response.ok)throw Error("HTTP "+response.status);return response.json();}
async function refresh(){
  const start=performance.now();
  const [snapshot,logs]=await Promise.allSettled([get("/api/snapshot"),get("/api/logs")]);
  try{
    if(snapshot.status==="fulfilled")render(snapshot.value);
    else{notice("连接中断，保留最近一次数据并自动重试。");set("status-text","连接中断");$("status").dataset.tone="failed";}
    if(logs.status==="fulfilled"){logData=logs.value;logProblem="";}else logProblem="日志连接中断，正在重试。";
    renderLog();
  }catch(e){notice("显示异常 · "+e.message);}
  finally{setTimeout(refresh,Math.max(0,1000-(performance.now()-start)));}
}
$("inspect-arm").addEventListener("change",readout);
$("compare-gen").addEventListener("change",compare);
["holdout-gen","receipt-gen","receipt-delay"].forEach(id=>$(id).addEventListener("change",audits));
["chart-arm","chart-scope"].forEach(id=>$(id).addEventListener("change",renderTrend));
$("chart-metric").addEventListener("change",()=>{metricChoices[category+"-"+$("chart-arm").value]=$("chart-metric").value;renderTrend();});
$("chart-tabs").addEventListener("click",e=>{const b=e.target.closest("[data-chart]");if(b){category=b.dataset.chart;renderTrend();}});
$("fit").addEventListener("click",()=>{fit=!fit;saveSetting("agi-plasticity-show-fit",String(fit));renderTrend();});
document.querySelectorAll("[data-stream]").forEach(b=>b.addEventListener("click",()=>{stream=b.dataset.stream;renderLog();}));
refresh();
