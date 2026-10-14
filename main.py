"""
KERRY 入库单 OCR 服务
- POST /api/ocr      : 上传图片(file)，返回结构化 JSON（供 TWMS 系统调用）
- GET  /             : 网页上传调试页
启动: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
from datetime import datetime
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from enhanced_ocr import recognize_enhanced
from export_excel import ExportRequest, build_excel
from label_counter import count_white_labels
from label_count_page import LABEL_COUNT_PAGE
from ocr_models import warmup_models
from parser import parse_batch_candidates

app = FastAPI(title="KERRY Invoice OCR", version="1.1")
_ocr_semaphore = asyncio.Semaphore(1)
_count_semaphore = asyncio.Semaphore(2)


@app.on_event("startup")
def warmup():
    warmup_models()


@app.post("/api/ocr")
async def api_ocr(
    file: UploadFile = File(...),
    batch_candidates: str = Form(""),
):
    """TWMS 调用入口：multipart/form-data，字段名 file，值为图片文件。"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件 (image/*)")
    raw = await file.read()
    try:
        parsed_batch_candidates = parse_batch_candidates(batch_candidates)
        async with _ocr_semaphore:
            data = await run_in_threadpool(
                recognize_enhanced,
                raw,
                parsed_batch_candidates,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"识别失败: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"识别失败: {e}")

    data["filename"] = file.filename
    return JSONResponse(data)


@app.post("/api/export")
def api_export(request: ExportRequest):
    workbook = build_excel(request)
    po_no = re.sub(r"[^A-Za-z0-9_-]", "", request.po_no)[:32] or "unknown"
    filename = f"invoice-{po_no}-{datetime.now():%Y%m%d-%H%M%S}.xlsx"
    return StreamingResponse(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/count-labels")
async def api_count_labels(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片不能超过 15 MB")
    try:
        async with _count_semaphore:
            result = await run_in_threadpool(count_white_labels, raw)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
def index():
    return HYBRID_PAGE


@app.get("/hybrid", response_class=HTMLResponse)
def hybrid():
    return HYBRID_PAGE


@app.get("/label-count", response_class=HTMLResponse)
def label_count():
    return LABEL_COUNT_PAGE


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASN 收货 - OCR 入库单</title>
<style>
  body{font-family:-apple-system,"Segoe UI",sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;color:#222;background:#f8fafc}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:#64748b;font-size:13px;margin-bottom:14px}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:12px 0}
  .header-info{display:flex;gap:24px;font-size:14px;margin-bottom:8px;flex-wrap:wrap}
  .header-info b{color:#0f172a}
  input[type=file]{margin:8px 0}
  button{border:0;border-radius:6px;padding:9px 18px;cursor:pointer;font-size:14px}
  .btn-ocr{background:#2563eb;color:#fff}
  .btn-confirm{background:#16a34a;color:#fff;font-size:15px;padding:11px 28px}
  button:disabled{background:#94a3b8;cursor:not-allowed}
  table{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}
  th,td{border:1px solid #e2e8f0;padding:4px 6px;text-align:left}
  th{background:#f1f5f9;position:sticky;top:0}
  td input{width:100%;border:1px solid transparent;background:transparent;font-size:12px;padding:3px}
  td input:focus{border-color:#2563eb;outline:none;background:#fff}
  tr.review{background:#fef9c3}       /* 需复核整行标黄 */
  td.low input{background:#fee2e2;color:#991b1b}  /* 低置信度单元格标红 */
  .conf{font-size:10px;color:#94a3b8;text-align:center}
  .conf.bad{color:#dc2626;font-weight:bold}
  .toolbar{display:flex;justify-content:space-between;align-items:center;margin-top:12px}
  .total{font-size:14px}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:6px}
  .badge.warn{background:#fef08a;color:#854d0e}
  .ok{color:#16a34a;font-size:13px}
</style>
</head>
<body>
<h1>ASN 收货</h1>
<div class="sub">上传 KERRY 入库单图片，自动识别后人工核实，一键收货</div>

<div class="card">
  <input type="file" id="f" accept="image/*">
  <button class="btn-ocr" id="btn" onclick="run()">识别图片</button>
  <span id="status" style="font-size:13px;color:#64748b;margin-left:10px"></span>
</div>

<div class="card">
  <span style="font-size:13px;color:#64748b">标注阈值（TWMS 可按业务配置）：</span>
  SKU< <input type="number" id="thSku" value="0.95" step="0.01" style="width:60px">
  Qty< <input type="number" id="thQty" value="0.90" step="0.01" style="width:60px">
  单元格< <input type="number" id="thCell" value="0.85" step="0.01" style="width:60px">
  <button onclick="renderTable()" style="background:#64748b;color:#fff">应用阈值</button>
</div>

<div class="card" id="out" style="display:none">
  <div class="header-info" id="hinfo"></div>
  <div style="overflow-x:auto;max-height:60vh;overflow-y:auto">
    <table id="tbl"></table>
  </div>
  <div class="toolbar">
    <div class="total">合计数量: <b id="totalQty">0</b>
      <span class="badge warn" id="reviewBadge" style="display:none"></span>
    </div>
    <button class="btn-confirm" id="confirmBtn" onclick="confirmReceive()">✓ 确认收货</button>
  </div>
</div>

<script>
const COLS=[
  {k:'sku',label:'SKU',w:130},
  {k:'name',label:'品名',w:220},
  {k:'qty',label:'Qty',w:60},
  {k:'exp',label:'EXP',w:110},
  {k:'mfg',label:'MFG',w:110},
  {k:'batch',label:'Batch',w:120},
  {k:'pallet',label:'Pallet',w:60},
  {k:'box',label:'Box',w:50},
];
let ROWS=[];

async function run(){
  const f=document.getElementById('f').files[0];
  if(!f){alert('请先选择图片');return;}
  const btn=document.getElementById('btn'),st=document.getElementById('status');
  btn.disabled=true;st.textContent='识别中…';
  const fd=new FormData();fd.append('file',f);
  try{
    const r=await fetch('/api/ocr',{method:'POST',body:fd});
    const d=await r.json();
    ROWS=d.rows||[];
    st.textContent='完成，共 '+d.row_count+' 行';
    document.getElementById('hinfo').innerHTML=
      '<span>PO号: <b>'+(d.header.po_no||'-')+'</b></span>'+
      '<span>入库日期: <b>'+(d.header.inbound_date||'-')+'</b></span>';
    renderTable();
    document.getElementById('out').style.display='block';
  }catch(e){st.textContent='出错: '+e;}
  finally{btn.disabled=false;}
}

function renderTable(){
  const thSku=parseFloat(document.getElementById('thSku').value)||0.95;
  const thQty=parseFloat(document.getElementById('thQty').value)||0.90;
  const thCell=parseFloat(document.getElementById('thCell').value)||0.85;
  let h='<tr><th>#</th>'+COLS.map(c=>'<th>'+c.label+'</th>').join('')+'<th>置信度</th></tr>';
  let reviewCount=0;
  ROWS.forEach((row,i)=>{
    const conf=row.confidence||{};
    // 阈值和标注由 TWMS 侧（前端）控制，OCR 只给原始置信度
    const rev=(conf.sku!==undefined && conf.sku<thSku)
           ||(conf.qty!==undefined && conf.qty<thQty)
           ||(row.row_confidence||1)<thCell;
    if(rev) reviewCount++;
    h+='<tr class="'+(rev?'review':'')+'">';
    h+='<td>'+(i+1)+'</td>';
    COLS.forEach(c=>{
      const cv=conf[c.k];
      const low=(cv!==undefined && cv<thCell);
      h+='<td class="'+(low?'low':'')+'"><input data-i="'+i+'" data-k="'+c.k+'" value="'+(row[c.k]||'')+'"></td>';
    });
    const rc=row.row_confidence||0;
    h+='<td class="conf '+((rc)<thCell?'bad':'')+'">'+(rc*100).toFixed(0)+'%</td>';
    h+='</tr>';
  });
  document.getElementById('tbl').innerHTML=h;
  bindInputs();updateTotal();
  const b=document.getElementById('reviewBadge');
  if(reviewCount>0){b.style.display='inline-block';b.textContent=reviewCount+' 行需复核';}else{b.style.display='none';}
}

function bindInputs(){
  document.querySelectorAll('#tbl input').forEach(inp=>{
    inp.onchange=e=>{
      const i=+e.target.dataset.i,k=e.target.dataset.k;
      ROWS[i][k]=e.target.value;
      updateTotal();
    };
  });
}

function updateTotal(){
  let t=0;
  ROWS.forEach(r=>{const q=parseInt(r.qty)||0;t+=q;});
  document.getElementById('totalQty').textContent=t;
}

function confirmReceive(){
  // 收集核实后的表格数据
  const data={
    po_no: document.querySelector('#hinfo b')?.textContent || '',
    items: ROWS.map(r=>({
      sku:r.sku, name:r.name, qty:parseInt(r.qty)||0,
      exp:r.exp, mfg:r.mfg, batch:r.batch, pallet:r.pallet, box:r.box
    }))
  };
  // TODO: 改成 TWMS 真实收货接口，例如:
  // fetch('/your/twms/receive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})
  alert('确认收货！共 '+data.items.length+' 行，合计 '+(data.items.reduce((s,x)=>s+x.qty,0))+' 件。\\n（此处改为 POST 到 TWMS 收货接口）');
  console.log('收货数据:',data);
}
</script>
</body>
</html>"""


HYBRID_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>入库单识别与核实</title>
<style>
  body{font-family:-apple-system,"Segoe UI",sans-serif;max-width:1400px;margin:20px auto;padding:0 16px;color:#222;background:#f8fafc}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:#64748b;font-size:13px;margin-bottom:14px}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:12px 0}
  button{border:0;border-radius:6px;padding:9px 18px;cursor:pointer;font-size:14px}
  .btn-ocr{background:#2563eb;color:#fff}
  .btn-export{background:#0f766e;color:#fff}
  button:disabled{background:#94a3b8;cursor:not-allowed}
  .table-wrap{overflow:auto;max-height:65vh;margin-top:10px}
  table{border-collapse:collapse;width:100%;min-width:1180px;font-size:12px;table-layout:fixed}
  th,td{border:1px solid #e2e8f0;padding:4px 6px;text-align:left}
  th{background:#f1f5f9;position:sticky;top:0}
  td input,td select{width:100%;border:1px solid transparent;background:transparent;font-size:12px;padding:3px}
  td input:focus,td select:focus{border-color:#2563eb;outline:none;background:#fff}
  td.manual input,td.manual select{background:#f8fafc;border-color:#cbd5e1}
  td.system input{background:#f1f5f9;border-color:#cbd5e1;color:#475569}
  td.low input,td.low select{background:#fef9c3;border-color:#eab308;color:#713f12}
  td.invalid input,td.invalid select{background:#ffedd5;border-color:#f97316;color:#9a3412}
  td.missing input,td.missing select{background:#fee2e2;border-color:#ef4444;color:#991b1b}
  tr.blocking td:first-child{border-left:3px solid #dc2626}
  .conf{font-size:10px;color:#94a3b8;text-align:center}
  .conf.bad{color:#dc2626;font-weight:bold}
  .header-row{display:flex;gap:12px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  .header-row label{font-size:13px;color:#475569}
  .header-row input{padding:5px 8px;border:1px solid #cbd5e1;border-radius:4px;font-size:13px}
  .issue-summary{display:flex;gap:16px;flex-wrap:wrap;color:#475569;font-size:12px;margin:8px 0}
  .issue-summary .missing-text{color:#b91c1c}
  .issue-summary .invalid-text{color:#c2410c}
  .issue-summary .low-text{color:#854d0e}
  .review-check{text-align:center;width:54px}
  .review-check input{width:18px;height:18px;cursor:pointer}
  .toolbar{display:flex;justify-content:space-between;align-items:center;margin-top:12px}
  .total{font-size:14px}
  .export-status{font-size:12px;color:#b91c1c;margin-left:auto;margin-right:12px}
  .page-head{display:flex;align-items:center;justify-content:space-between;gap:12px}
  .tool-link{font-size:13px;color:#0f766e;text-decoration:none}
</style>
</head>
<body>
<div class="page-head"><h1>入库单识别与核实</h1><a class="tool-link" href="/label-count">面单计数</a></div>

<div class="card">
  <input type="file" id="f" accept="image/*">
  <label style="font-size:13px;color:#475569;margin:0 8px">批次号候选 <input id="m_batches" style="width:300px;padding:5px 8px;border:1px solid #cbd5e1;border-radius:4px" placeholder="多个批次号用逗号分隔"></label>
  <button class="btn-ocr" id="btn" onclick="run()">识别图片</button>
  <span id="status" style="font-size:13px;color:#64748b;margin-left:10px"></span>
</div>

<div class="card" id="out" style="display:none">
  <div class="header-row">
    <label>PO 号 <input id="m_po" style="width:140px"></label>
    <label>备注 <input id="m_note" style="width:200px" placeholder="手写备注"></label>
  </div>
  <div class="issue-summary" id="issueSummary"></div>
  <div class="table-wrap" id="tableWrap">
    <table id="tbl"></table>
  </div>
</div>

<div class="card" id="exportArea" style="display:none">
  <div class="toolbar">
    <div class="total">明细合计: <b id="totalQty">0</b> / <span id="documentTotalLabel">单据合计</span>: <b id="documentTotal">-</b></div>
    <span class="export-status" id="exportStatus" role="status"></span>
    <button class="btn-export" id="exportBtn" onclick="exportExcel()">导出 Excel</button>
  </div>
</div>

<script>
const COLS=[
  {k:'sku',label:'SKU',w:130,required:true},
  {k:'name',label:'品名',w:200,system:true},
  {k:'qty',label:'Qty',w:60,required:true},
  {k:'exp',label:'EXP',w:100},
  {k:'mfg',label:'MFG',w:100},
  {k:'batch',label:'批次号',w:140,manual:true},
  {k:'pallet',label:'板号',w:80,manual:true},
  {k:'box',label:'箱号',w:60,manual:true},
];
const CONFIDENCE_THRESHOLD=0.85;
let ROWS=[];
let DOCUMENT_TOTAL=null;
let ACTIVE_COLS=COLS;
let PAGE_NUMBER=null;
let TOTAL_PAGES=null;
let BATCH_CANDIDATES=[];

function escapeHtml(value){
  return String(value??'').replace(/[&<>"']/g,ch=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
  })[ch]);
}

function normalizeBatch(value){
  return String(value??'').toUpperCase().replace(/[^A-Z0-9]/g,'');
}

function parseBatchCandidates(){
  const values=document.getElementById('m_batches').value
    .split(/[\\s,，;；]+/)
    .map(value=>value.trim())
    .filter(Boolean);
  const unique=new Map();
  values.forEach(value=>{const key=normalizeBatch(value);if(key&&!unique.has(key))unique.set(key,value);});
  return Array.from(unique.values());
}

function editDistance(left,right){
  const previous=Array.from({length:right.length+1},(_,index)=>index);
  for(let i=1;i<=left.length;i++){
    let diagonal=previous[0];previous[0]=i;
    for(let j=1;j<=right.length;j++){
      const old=previous[j];
      previous[j]=Math.min(previous[j]+1,previous[j-1]+1,diagonal+(left[i-1]===right[j-1]?0:1));
      diagonal=old;
    }
  }
  return previous[right.length];
}

function matchBatchCandidate(rawValue){
  const raw=normalizeBatch(rawValue);
  if(!raw||!BATCH_CANDIDATES.length)return '';
  const scored=BATCH_CANDIDATES.map(value=>({
    value:value,
    distance:editDistance(raw,normalizeBatch(value)),
  })).sort((a,b)=>a.distance-b.distance);
  const threshold=Math.max(1,Math.floor(raw.length*0.25));
  const uniqueBest=scored.length===1||scored[0].distance<scored[1].distance;
  return uniqueBest&&scored[0].distance<=threshold?scored[0].value:'';
}

function applyBatchCandidates(shouldRender=true){
  BATCH_CANDIDATES=parseBatchCandidates();
  ROWS.forEach(row=>{
    row._rawBatch=row._rawBatch??String(row.batch??'');
    if((row._edited||{}).batch)return;
    row.batch=BATCH_CANDIDATES.length?matchBatchCandidate(row._rawBatch):row._rawBatch;
  });
  if(shouldRender)renderTable();
}

function isValidDate(value){
  if(!value)return true;
  const m=value.match(/^([0-9]{4})-([0-9]{2})-([0-9]{2})$/);
  if(!m)return false;
  const year=Number(m[1]),month=Number(m[2]),day=Number(m[3]);
  if(year<1900||year>2200||month<1||month>12)return false;
  return day>=1&&day<=new Date(year,month,0).getDate();
}

function getFieldIssue(row,col){
  if(col.system)return '';
  const value=String(row[col.k]??'').trim();
  if(col.required&&!value)return 'missing';
  if(col.k==='sku'&&value&&!/^[0-9]{8,}$/.test(value))return 'invalid';
  if(col.k==='qty'&&value&&!/^[1-9][0-9]*$/.test(value))return 'invalid';
  if((col.k==='exp'||col.k==='mfg')&&value&&!isValidDate(value))return 'invalid';
  const backendError=(row.validation_errors||{})[col.k];
  if(backendError==='missing')return 'missing';
  if(backendError&&!((row._edited||{})[col.k]))return 'invalid';
  const rawScore=(row.confidence||{})[col.k];
  const score=rawScore===null||rawScore===undefined?NaN:Number(rawScore);
  if(Number.isFinite(score)&&score<CONFIDENCE_THRESHOLD&&!((row._edited||{})[col.k])&&!row._reviewed)return 'low';
  return '';
}

function getCriticalIssues(row){
  return COLS.filter(col=>col.required)
    .map(col=>({col,issue:getFieldIssue(row,col)}))
    .filter(item=>item.issue);
}

function hasHardCriticalIssue(row){
  return getCriticalIssues(row).some(item=>item.issue==='missing'||item.issue==='invalid');
}

function needsConfidenceReview(row){
  return COLS.filter(col=>col.required).some(col=>{
    const rawScore=(row.confidence||{})[col.k];
    const score=rawScore===null||rawScore===undefined?NaN:Number(rawScore);
    return Number.isFinite(score)&&score<CONFIDENCE_THRESHOLD&&!((row._edited||{})[col.k]);
  });
}

async function run(){
  const f=document.getElementById('f').files[0];
  if(!f){alert('请先选择图片');return;}
  const btn=document.getElementById('btn'),st=document.getElementById('status');
  btn.disabled=true;st.textContent='识别中…';
  BATCH_CANDIDATES=parseBatchCandidates();
  const fd=new FormData();fd.append('file',f);
  fd.append('batch_candidates',JSON.stringify(BATCH_CANDIDATES));
  try{
    const r=await fetch('/api/ocr',{method:'POST',body:fd});
    const d=await r.json().catch(()=>({}));
    if(!r.ok)throw new Error(d.detail||('识别请求失败（'+r.status+'）'));
    BATCH_CANDIDATES=Array.isArray(d.batch_candidates)?d.batch_candidates:BATCH_CANDIDATES;
    ROWS=(d.rows||[]).map(row=>({...row,_reviewed:false,_edited:{},_rawBatch:String(row.raw_batch??row.batch??'')}));
    applyBatchCandidates(false);
    DOCUMENT_TOTAL=Number.isInteger(d.document_total_qty)?d.document_total_qty:null;
    PAGE_NUMBER=Number.isInteger((d.header||{}).page_number)?d.header.page_number:null;
    TOTAL_PAGES=Number.isInteger((d.header||{}).total_pages)?d.header.total_pages:null;
    ACTIVE_COLS=(d.image||{}).table_layout==='without_batch'
      ?COLS.filter(col=>col.k!=='batch')
      :COLS;
    const pageText=TOTAL_PAGES>1?' · 第 '+PAGE_NUMBER+'/'+TOTAL_PAGES+' 页':'';
    st.textContent='完成，共 '+d.row_count+' 行'+pageText+' · '+((d.models||{}).general||'PP-OCRv6-small');
    document.getElementById('m_po').value=(d.header||{}).po_no||'';
    document.getElementById('m_note').value=d.notes||'';
    renderTable();
    document.getElementById('out').style.display='block';
    document.getElementById('exportArea').style.display='block';
  }catch(e){st.textContent='出错: '+e.message;}
  finally{btn.disabled=false;}
}

function renderTable(){
  const wrap=document.getElementById('tableWrap');
  const scrollTop=wrap.scrollTop;
  let h='<colgroup><col style="width:36px">'+ACTIVE_COLS.map(c=>'<col style="width:'+c.w+'px">').join('')+'<col style="width:64px"><col style="width:54px"></colgroup>';
  h+='<thead><tr><th>#</th>'+ACTIVE_COLS.map(c=>'<th>'+c.label+'</th>').join('')+'<th>置信度</th><th>核实</th></tr></thead><tbody>';
  ROWS.forEach((row,i)=>{
    const rc=Number(row.row_confidence)||0;
    const blocking=getCriticalIssues(row).length>0;
    h+='<tr class="'+(blocking?'blocking':'')+'"><td>'+(i+1)+'</td>';
    ACTIVE_COLS.forEach(c=>{
      const issue=getFieldIssue(row,c);
      const classes=[c.manual?'manual':'',c.system?'system':'',issue].filter(Boolean).join(' ');
      const title=issue==='missing'?'必填缺失':issue==='invalid'?'格式异常':issue==='low'?'低置信度':'';
      const placeholder=c.system?'系统查询':c.manual?'手动填':'';
      if(c.k==='batch'&&BATCH_CANDIDATES.length){
        const current=String(row.batch??'');
        const options=[''].concat(BATCH_CANDIDATES);
        if(current&&!options.includes(current))options.push(current);
        h+='<td class="'+classes+'"><select data-i="'+i+'" data-k="batch" title="'+title+'">'+options.map(value=>'<option value="'+escapeHtml(value)+'" '+(value===current?'selected':'')+'>'+(value?escapeHtml(value):'手动选择')+'</option>').join('')+'</select></td>';
      }else{
        h+='<td class="'+classes+'"><input data-i="'+i+'" data-k="'+c.k+'" value="'+escapeHtml(row[c.k]||'')+'" placeholder="'+placeholder+'" title="'+title+'" '+(c.system?'readonly':'')+'></td>';
      }
    });
    h+='<td class="conf '+(rc<CONFIDENCE_THRESHOLD?'bad':'')+'">'+(rc*100).toFixed(0)+'%</td>';
    const hard=hasHardCriticalIssue(row),needsReview=needsConfidenceReview(row);
    const checked=!hard&&(!needsReview||row._reviewed);
    const disabled=hard||!needsReview;
    h+='<td class="review-check"><input type="checkbox" aria-label="核实第'+(i+1)+'行" onchange="setRowReviewed('+i+',this.checked)" '+(checked?'checked ':'')+(disabled?'disabled':'')+'></td></tr>';
  });
  h+='</tbody>';
  document.getElementById('tbl').innerHTML=h;
  bindInputs();updateValidation();
  wrap.scrollTop=scrollTop;
}

function bindInputs(){
  document.querySelectorAll('#tbl [data-k]').forEach(inp=>{
    inp.onchange=e=>{
      const i=+e.target.dataset.i,k=e.target.dataset.k;
      ROWS[i][k]=e.target.value;
      ROWS[i]._edited[k]=true;
      ROWS[i]._reviewed=false;
      renderTable();
    };
  });
}

function setRowReviewed(index,checked){
  if(hasHardCriticalIssue(ROWS[index]))return;
  ROWS[index]._reviewed=checked;
  renderTable();
}

function updateTotal(){
  let total=0,allValid=true;
  ROWS.forEach(row=>{
    const qty=String(row.qty??'').trim();
    if(/^[1-9][0-9]*$/.test(qty))total+=Number(qty);
    else allValid=false;
  });
  document.getElementById('totalQty').textContent=allValid?total:'待核实';
  document.getElementById('documentTotal').textContent=DOCUMENT_TOTAL??'-';
  document.getElementById('documentTotalLabel').textContent=TOTAL_PAGES>1?'整单合计':'单据合计';
  return total;
}

function updateValidation(){
  const counts={missing:0,invalid:0,low:0};
  ROWS.forEach(row=>ACTIVE_COLS.forEach(col=>{
    const issue=getFieldIssue(row,col);if(issue)counts[issue]++;
  }));
  const pendingRows=ROWS.filter(row=>getCriticalIssues(row).length>0).length;
  document.getElementById('issueSummary').innerHTML=
    '<span class="missing-text">必填缺失 '+counts.missing+'</span>'+
    '<span class="invalid-text">格式异常 '+counts.invalid+'</span>'+
    '<span class="low-text">低置信度 '+counts.low+'</span>'+
    '<span>待核实行 '+pendingRows+'</span>';

  updateTotal();
  document.getElementById('exportBtn').disabled=ROWS.length===0;
}

async function exportExcel(){
  if(!ROWS.length)return;
  const btn=document.getElementById('exportBtn'),status=document.getElementById('exportStatus');
  btn.disabled=true;status.textContent='';
  const payload={
    po_no:document.getElementById('m_po').value,
    note:document.getElementById('m_note').value,
    document_total:DOCUMENT_TOTAL,
    include_batch:ACTIVE_COLS.some(col=>col.k==='batch'),
    rows:ROWS.map(row=>{
      const confidence=Number(row.row_confidence);
      return {
        sku:String(row.sku??''),name:String(row.name??''),qty:String(row.qty??''),
        exp:String(row.exp??''),mfg:String(row.mfg??''),batch:String(row.batch??''),
        pallet:String(row.pallet??''),box:String(row.box??''),
        row_confidence:row.row_confidence==null||!Number.isFinite(confidence)?null:confidence,
        reviewed:!hasHardCriticalIssue(row)&&(!needsConfidenceReview(row)||row._reviewed),
      };
    }),
  };
  try{
    const response=await fetch('/api/export',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),
    });
    if(!response.ok)throw new Error('导出失败（'+response.status+'）');
    const blob=await response.blob();
    const link=document.createElement('a'),url=URL.createObjectURL(blob);
    const disposition=response.headers.get('Content-Disposition')||'';
    const match=disposition.match(/filename="([^"]+)"/);
    link.href=url;
    link.download=match?match[1]:'invoice.xlsx';
    document.body.append(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){status.textContent=error.message;}
  finally{btn.disabled=false;}
}

document.getElementById('m_batches').addEventListener('change',()=>applyBatchCandidates(true));
</script>
</body>
</html>"""
