# Package Text OCR API

This is a separate lightweight service for package production dates, expiry dates, and batch
numbers. It does not import the invoice table pipeline, spreadsheet export, or identifier model.

## Start

Create a local `.env` file from `package.env.example`, then run:

```bash
docker compose --env-file .env -f docker-compose.package.yml up -d --build
```

The container port is bound to host loopback only: `127.0.0.1:8005`. TWMS should call:

```text
POST http://127.0.0.1:8005/api/package-text
X-OCR-Token: <shared internal token>
```

Multipart fields:

- `file`: JPEG or PNG crop, at most 1 MiB by default.
- `required_fields`: JSON array containing `manufacture_at`, `expire_at`, and/or `batch`.
- `batch_candidates`: optional JSON array from the selected ASN lines.

Images are decoded in memory and are not written to disk. The first recognition uses the original
crop. A single contrast-enhanced retry runs only when a required field is missing.

## Response

```json
{
  "manufacture_at": "2026-04-01",
  "expire_at": "2028-03-21",
  "batch": "61840E316B",
  "batch_original": "618405316B",
  "batch_correction": "asn_batch_candidate",
  "raw_text": "...",
  "missing_required_fields": [],
  "batch_review_positions": [5],
  "timing_ms": {"decode": 5, "primary_ocr": 420, "enhanced_ocr": 0, "total": 426},
  "model": "PP-OCRv6-small",
  "request_id": "..."
}
```

TWMS remains the only PDA-facing endpoint. Do not expose this container port publicly.

## 通用文字与日期识别

同一服务新增 `POST /api/text`，复用认证、上传限制、模型与并发控制。
上传 `file` 即可，不需要指定批次或日期字段。可选 `date_order` 为 `auto`（默认自动判断）、
`dmy`（日月年）或 `mdy`（月日年）。手持终端通过已认证的 TWMS（仓储系统）
`inbound/text/recognize` 接口转发，不能直接连接识别服务。

返回 `raw_text`（原文）、`normalized_text`（明确日期已标准化的文本）、
`lines`（按阅读顺序的逐行文字和置信度）、`dates`（日期解析结果）。
日期结果包含 `raw`、`value`、`candidates`、`status`、`field` 与 `line_index`。
`status` 分为 `valid`（明确）、`ambiguous`（歧义）、`invalid`（日期无效）。
歧义或无效日期保留原文，不强制选择候选；无标签日期不推断为生产日期或效期。

支持四位年份的年月日、日月年／月日年数字日期、中文年月日、英文月份日期、
独立八位年月日／日月年／月日年（歧义时保留候选）。月份年份等不完整日期和两位年份不自动补齐。
`01/02/2028` 在自动模式下返回两个候选；`31/12/2028` 标准化为 `2028-12-31`。

### 单号快速识别与耗时

`/api/text` 支持可选字段 `text_mode`：`paragraph`（默认段落）、`line`（通用单行）、
`identifier`（单行单号）。旧请求省略字段时沿用完整检测流程。

签收单号入口的单行模式使用 `identifier`：先直接识别整行，只有单个、不含空白、包含数字的
标识且置信度至少 0.95 时接受结果；其他情况回退完整检测。普通句子和中文使用完整流程，
避免直接行识别漏掉单词间空格。段落模式同样使用完整流程。识别结果仍须人工核对。

`timing_ms` 包含 `decode`、`line_ocr`、`primary_ocr`、`enhanced_ocr` 和 `total`。
这些时间不包含请求在服务入口前的网络传输或等待并发槽位的时间。
仓储系统响应另含 `proxy_timing_ms`，手持终端日志标签 `TextRecognitionTiming` 记录
拍照、裁剪、图片字节数、请求及上传写入时间。上传写入时间不等于完整网络传输时间。

真实模型合成样例校验：`python tests/benchmark_single_line.py --font <中文字体路径> --runs 3`。
该命令只生成测试图片，不读取客户图片；分别验证单号、带空格英文、日期及中文，在四组明暗背景下
核对文本和日期解析一致性，并输出本机模型耗时。模型耗时不能直接当作线上整次拍照识别耗时。

发布顺序为识别服务、仓储系统后端、手持终端。旧服务会忽略新增字段并沿用完整流程；
只有三端都更新后，签收单号快速路径才会生效。
连续英文或数字编码中嵌入的日期数字不作为独立日期提取。
批次号、商品编码等明确标记的值也不会被改成日期；日期用途仅使用当前日期前紧邻的标签，
不会沿用前一个日期的生产／效期标签。支持标签与日期紧连，例如 `EXP2028/12/31`。

模型支持白底黑字和黑底白字，不要求操作员手工反色。真实模型验证可运行
`python tests/benchmark_text_polarity.py`，覆盖白底黑字、黑底白字、深灰底浅色字和黑底灰字，
同时验证通用文字与原批次／效期接口。模型目录可通过 `RAPIDOCR_MODEL_ROOT_DIR` 环境变量指定。

手持终端在收货属性弹窗新增“通用文字识别”，沿用原文字采集的启用条件。
可选单行或段落取景；识别结果可编辑、复制，日期结果单独展示，且不自动写入收货属性。
通用接口代理超时由 `PACKAGE_TEXT_OCR_TEXT_TIMEOUT`（默认 15 秒）控制；
原包装字段接口仍使用原有超时。发布时先更新识别服务，再更新仓储后端，最后更新手持终端。
