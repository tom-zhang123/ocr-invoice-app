"""Standalone camera UI for cumulative white-label counting."""


LABEL_COUNT_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>面单计数</title>
<style>
  :root{color-scheme:light;--ink:#17212b;--muted:#64748b;--line:#dbe3ea;--panel:#fff;--bg:#f4f7f9;--accent:#087f75;--accent-dark:#066b63;--warn:#b45309}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;letter-spacing:0}
  button,input{font:inherit;letter-spacing:0}
  button{border:0;cursor:pointer}
  button:disabled{cursor:not-allowed;opacity:.45}
  .shell{width:min(1180px,100%);margin:0 auto;padding:16px}
  .topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}
  h1{font-size:22px;line-height:1.2;margin:0}
  .back{color:#475569;text-decoration:none;font-size:14px}
  .workspace{display:grid;grid-template-columns:minmax(0,1fr) 320px;border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden;min-height:610px}
  .capture-pane{padding:14px;border-right:1px solid var(--line)}
  .camera-stage{position:relative;width:100%;aspect-ratio:4/3;background:#111827;overflow:hidden}
  video,canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#111827}
  canvas{display:none}
  .empty-state{position:absolute;inset:0;display:grid;place-items:center;color:#cbd5e1;font-size:14px;text-align:center;padding:24px}
  .empty-state[hidden]{display:none}
  .capture-bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:12px}
  .primary,.secondary,.danger,.file-label{height:40px;padding:0 16px;border-radius:6px;display:inline-flex;align-items:center;justify-content:center;white-space:nowrap}
  .primary{background:var(--accent);color:#fff;font-weight:600}
  .primary:hover{background:var(--accent-dark)}
  .secondary,.file-label{background:#e8eef2;color:#263645}
  .danger{background:#fff1f2;color:#9f1239;border:1px solid #fecdd3}
  .file-label{cursor:pointer}
  .file-label input{display:none}
  .status{min-height:20px;color:var(--muted);font-size:13px;flex:1 1 180px}
  .status.error{color:#b91c1c}
  .side{padding:16px;display:flex;flex-direction:column;min-width:0}
  .metrics{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line);padding-bottom:16px}
  .metric{min-width:0}
  .metric+.metric{border-left:1px solid var(--line);padding-left:16px}
  .metric-label{font-size:12px;color:var(--muted);margin-bottom:4px}
  .metric-value{font-size:40px;line-height:1;font-variant-numeric:tabular-nums;font-weight:700}
  .adjust{display:flex;align-items:center;gap:8px;margin-top:10px}
  .icon-button{width:34px;height:34px;border:1px solid #cbd5e1;background:#fff;color:#334155;font-size:22px;line-height:1;border-radius:4px}
  .history-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:18px 0 8px}
  .history-head h2{font-size:15px;margin:0}
  .history-actions{display:flex;gap:6px}
  .history-actions button{background:transparent;color:#475569;padding:5px 7px;border-radius:4px}
  .history{width:100%;border-collapse:collapse;font-size:13px}
  .history th,.history td{padding:8px 6px;border-bottom:1px solid #edf1f4;text-align:right;font-variant-numeric:tabular-nums}
  .history th:first-child,.history td:first-child{text-align:left}
  .history th{color:var(--muted);font-weight:500}
  .history-empty{color:#94a3b8;text-align:center;padding:26px 0;font-size:13px}
  .pulse{animation:pulse .7s ease-out}
  @keyframes pulse{0%{color:var(--accent);transform:scale(1.08)}100%{color:var(--ink);transform:scale(1)}}
  @media(max-width:760px){
    .shell{padding:10px}
    .workspace{grid-template-columns:1fr;min-height:0}
    .capture-pane{border-right:0;border-bottom:1px solid var(--line);padding:10px}
    .side{padding:14px}
    .camera-stage{aspect-ratio:3/4;max-height:62vh}
    .capture-bar{display:grid;grid-template-columns:1fr 1fr}
    .capture-bar .primary{grid-column:1/-1;height:48px}
    .capture-bar .status{grid-column:1/-1}
    .metric-value{font-size:34px}
  }
</style>
</head>
<body>
<main class="shell">
  <header class="topbar">
    <h1>面单计数</h1>
    <a class="back" href="/hybrid">入库单识别</a>
  </header>
  <section class="workspace">
    <div class="capture-pane">
      <div class="camera-stage" id="stage">
        <video id="video" playsinline muted></video>
        <canvas id="canvas"></canvas>
        <div class="empty-state" id="emptyState">摄像头未开启</div>
      </div>
      <div class="capture-bar">
        <button class="secondary" id="startBtn" type="button">开启摄像头</button>
        <button class="primary" id="captureBtn" type="button" disabled>拍照计数</button>
        <label class="file-label">选择照片<input id="photoInput" type="file" accept="image/*" capture="environment"></label>
        <span class="status" id="status" role="status"></span>
      </div>
    </div>
    <aside class="side">
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">本次数量</div>
          <div class="metric-value" id="currentCount">0</div>
          <div class="adjust">
            <button class="icon-button" id="minusBtn" type="button" title="本次数量减一" disabled>&minus;</button>
            <button class="icon-button" id="plusBtn" type="button" title="本次数量加一" disabled>+</button>
          </div>
        </div>
        <div class="metric">
          <div class="metric-label">累计数量</div>
          <div class="metric-value" id="totalCount">0</div>
        </div>
      </div>
      <div class="history-head">
        <h2>拍照记录</h2>
        <div class="history-actions">
          <button id="undoBtn" type="button" disabled>撤销上一笔</button>
          <button id="resetBtn" type="button" disabled>清零</button>
        </div>
      </div>
      <div id="historyEmpty" class="history-empty">暂无记录</div>
      <table class="history" id="historyTable" hidden>
        <thead><tr><th>次数</th><th>本次</th><th>累计</th></tr></thead>
        <tbody id="historyBody"></tbody>
      </table>
    </aside>
  </section>
</main>
<script>
const video=document.getElementById('video');
const canvas=document.getElementById('canvas');
const context=canvas.getContext('2d');
const startBtn=document.getElementById('startBtn');
const captureBtn=document.getElementById('captureBtn');
const photoInput=document.getElementById('photoInput');
const statusEl=document.getElementById('status');
const emptyState=document.getElementById('emptyState');
const currentEl=document.getElementById('currentCount');
const totalEl=document.getElementById('totalCount');
const minusBtn=document.getElementById('minusBtn');
const plusBtn=document.getElementById('plusBtn');
const undoBtn=document.getElementById('undoBtn');
const resetBtn=document.getElementById('resetBtn');
const historyTable=document.getElementById('historyTable');
const historyEmpty=document.getElementById('historyEmpty');
const historyBody=document.getElementById('historyBody');
let stream=null,busy=false,shots=[],resumeTimer=null;

function setStatus(message,error=false){statusEl.textContent=message;statusEl.classList.toggle('error',error);}
function total(){return shots.reduce((sum,shot)=>sum+shot.count,0);}
function render(){
  const current=shots.length?shots[shots.length-1].count:0;
  currentEl.textContent=current;totalEl.textContent=total();
  currentEl.classList.remove('pulse');totalEl.classList.remove('pulse');
  void totalEl.offsetWidth;
  if(shots.length){currentEl.classList.add('pulse');totalEl.classList.add('pulse');}
  minusBtn.disabled=!shots.length||current<=0;plusBtn.disabled=!shots.length;
  undoBtn.disabled=!shots.length;resetBtn.disabled=!shots.length;
  historyEmpty.hidden=shots.length>0;historyTable.hidden=!shots.length;
  let running=0;
  historyBody.innerHTML=shots.map((shot,index)=>{
    running+=shot.count;
    return '<tr><td>第 '+(index+1)+' 次</td><td>'+shot.count+'</td><td>'+running+'</td></tr>';
  }).join('');
}

async function startCamera(){
  if(stream)return;
  if(!navigator.mediaDevices?.getUserMedia){setStatus('当前浏览器无法调用摄像头',true);return;}
  startBtn.disabled=true;setStatus('正在开启摄像头…');
  try{
    stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false,
    });
    video.srcObject=stream;await video.play();
    emptyState.hidden=true;video.style.display='block';canvas.style.display='none';
    captureBtn.disabled=false;startBtn.textContent='摄像头已开启';setStatus('');
  }catch(error){
    setStatus('无法开启摄像头，请检查浏览器权限或使用 HTTPS',true);
    startBtn.disabled=false;
  }
}

function drawResult(result){
  context.lineWidth=Math.max(3,canvas.width/300);
  context.strokeStyle='#16a34a';context.fillStyle='#16a34a';
  context.font='bold '+Math.max(18,canvas.width/40)+'px Arial';
  result.boxes.forEach((box,index)=>{
    context.strokeRect(box.x,box.y,box.width,box.height);
    const label=String(index+1),padding=6;
    const width=context.measureText(label).width+padding*2;
    context.fillRect(box.x,Math.max(0,box.y-30),width,30);
    context.fillStyle='#fff';context.fillText(label,box.x+padding,Math.max(22,box.y-7));context.fillStyle='#16a34a';
  });
  video.style.display='none';canvas.style.display='block';emptyState.hidden=true;
  clearTimeout(resumeTimer);
  if(stream)resumeTimer=setTimeout(()=>{canvas.style.display='none';video.style.display='block';},1400);
}

async function analyze(blob){
  if(busy)return;
  busy=true;captureBtn.disabled=true;photoInput.disabled=true;setStatus('正在计数…');
  const form=new FormData();form.append('file',blob,'capture.jpg');
  try{
    const response=await fetch('/api/count-labels',{method:'POST',body:form});
    const result=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error(result.detail||('计数失败（'+response.status+'）'));
    drawResult(result);shots.push({count:result.count});render();setStatus('本次识别 '+result.count+' 张');
  }catch(error){setStatus(error.message,true);}
  finally{busy=false;captureBtn.disabled=!stream;photoInput.disabled=false;}
}

function capture(){
  if(!stream||video.readyState<2||busy)return;
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  context.drawImage(video,0,0,canvas.width,canvas.height);
  canvas.toBlob(blob=>{if(blob)analyze(blob);else setStatus('拍照失败，请重试',true);},'image/jpeg',.92);
}

function loadPhoto(file){
  if(!file||busy)return;
  const image=new Image(),url=URL.createObjectURL(file);
  image.onload=()=>{
    canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    context.drawImage(image,0,0);URL.revokeObjectURL(url);analyze(file);
  };
  image.onerror=()=>{URL.revokeObjectURL(url);setStatus('图片无法读取',true);};
  image.src=url;
}

startBtn.addEventListener('click',startCamera);
captureBtn.addEventListener('click',capture);
photoInput.addEventListener('change',event=>{loadPhoto(event.target.files[0]);event.target.value='';});
minusBtn.addEventListener('click',()=>{if(shots.length&&shots.at(-1).count>0){shots.at(-1).count--;render();}});
plusBtn.addEventListener('click',()=>{if(shots.length){shots.at(-1).count++;render();}});
undoBtn.addEventListener('click',()=>{if(shots.length){shots.pop();render();setStatus('已撤销上一笔');}});
resetBtn.addEventListener('click',()=>{shots=[];render();setStatus('累计已清零');});
window.addEventListener('pagehide',()=>{stream?.getTracks().forEach(track=>track.stop());});
render();
</script>
</body>
</html>"""
