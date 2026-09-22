# OCR Invoice App

独立的 KERRY 入库单 OCR（文字识别）服务。项目不依赖 TWMS（仓储管理系统）源码、数据库或运行容器，可单独开发、测试、训练和发布。

## 当前能力

- PP-OCRv6-small（第六版小型模型）负责整页和单元格识别
- 自动校正拍照透视并检测 7 列、8 列和续页表格
- 品名不进入 OCR（文字识别），由调用系统按 SKU（商品编码）查询
- 默认执行快速路径，只对缺失、格式异常或低置信度字段补识别
- 支持随图片提交允许批次号，并做唯一近似匹配
- 日期不满足完整 `YYYY-MM-DD` 时置空，交由操作员录入
- 返回字段置信度、校验错误、纠正原因、执行路径和分阶段耗时

当前五张真值样图共 155 行，数量逐行命中 153/155（98.7%），行数全部正确。本机平均约 6.96 秒/张；2 核 2GB 服务器的标准样图约 15.6 秒，复杂样图约 29.1 秒。耗时会随 CPU（处理器）、内存、图片尺寸和补识别字段变化。

## 项目结构

```text
.
|-- main.py                  FastAPI（接口框架）入口和核实页面
|-- enhanced_ocr.py          表格定位、快速路径和失败补识别
|-- parser.py                字段清洗、校验和保守纠正
|-- ocr_models.py            PP-OCRv6-small（第六版小型模型）配置
|-- tests/                   单元测试和离线基准脚本
|-- training/                训练数据规范和后续微调说明
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
```

客户图片、训练集、自定义模型、训练产物和发布包均被 Git（版本管理）忽略，不应提交到仓库。

## Docker（容器引擎）启动

```powershell
cd D:\dev\ocr-invoice-app
docker-compose up -d --build
```

默认地址：

- 核实页面：<http://127.0.0.1:8004/hybrid>
- 接口说明：<http://127.0.0.1:8004/docs>
- 识别接口：`POST http://127.0.0.1:8004/api/ocr`

如需更换端口，修改 `docker-compose.yml` 中的 `8004:8000`。

## 接口调用

```bash
curl -X POST http://127.0.0.1:8004/api/ocr \
  -F "file=@invoice.jpg" \
  -F "batch_candidates=628B18ST01,116F985T02"
```

`batch_candidates` 可以为空。为空时保留原始批次识别逻辑。

主要返回字段：

```json
{
  "header": {"po_no": "PO000001", "page_number": 1, "total_pages": 1},
  "row_count": 2,
  "document_total_qty": 28,
  "pipeline": {
    "name_column_skipped": true,
    "sku_fallback": false,
    "handwriting_fallback_fields": ["qty", "exp", "mfg"]
  },
  "timing_ms": {"page_ocr": 2800, "structured_cells": 900, "total": 6200},
  "rows": []
}
```

## 本机直接运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8004
```

## 测试

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

模型基准使用 `tests/benchmark_model.py`，图片路径和人工真值必须在执行时显式传入，仓库不保存客户样图。

## 训练与模型发布

当前服务不会在线自主学习。操作员确认后的图片和正确字段应先保存为版本化训练集，再离线微调 Recognition（文字识别）模型。训练格式、数据拆分、验收和发布约束见 [training/README.md](training/README.md)。

训练生成的新模型不能直接覆盖线上模型。必须先通过独立验证集，生成带版本号的模型包，再人工切换并保留回滚版本。

## 已知边界

- 手写日期、批次、板号和箱号仍可能需要人工核实
- 多页单据需要由调用系统合并全部页面后再确认
- 无法唯一推断的值不会用宽松规则强行改写
- 公网部署应增加 IP（网络地址）白名单、鉴权、上传大小限制和请求频率限制
