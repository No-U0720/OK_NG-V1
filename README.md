Markdown

# 🔍 接腳長度自動檢測系統 (Prong Length Detection Tool)

這是一個基於 Python、OpenCV 與 AI 去背技術 (Rembg) 開發的 **AOI 自動光學檢測系統**。系統能自動辨識金屬/塑膠零件的左右接腳（Prongs），進行幾何測量並判定長度是否合格（OK / NG）。

專案同時提供 **命令列工具 (CLI)** 與 **現代化網頁介面 (Web UI)**，方便產線自動化批次處理或人員互動操作。

---

## 🌟 核心特色

- 🤖 **AI 智慧背景去除**：整合 `rembg` 庫自動去背，搭配動態 Alpha 遮罩與灰階閾值，完美隔絕背景雜訊與陰影干擾。
- 📐 **幾何線段精確量測**：
  - 自動定位接腳頂點（Tip Point）與兩側基部點（Base Points）。
  - 建立基線並計算垂直投影距離（Perpendicular Distance），實時測量接腳實際像素長度（$L$）。
- 🎨 **視覺化標記與原因繪製**：自動於輸出影像標記綠色輪廓、基準線、幾何關鍵點與數據標籤（支援中文字體渲染）。
- 🌐 **內建 Web UI**：採用零外部依賴的 Python `http.server` 打造，支援拖曳上傳與即時檢測結果展示。
- 📁 **多模式執行**：支援單張圖片檢測、資料夾批次掃描以及 Web 互動模式。

---

## 🛠️ 環境需求與安裝

請確保您的環境已安裝 **Python 3.8+**。

### 1. 複製專案與下載
```bash
git clone [https://github.com/No-U0720/OK_NG.git](https://github.com/No-U0720/OK_NG.git)
cd OK_NG

2. 安裝依賴套件
Bash

pip install numpy opencv-python pillow rembg

    💡 提示：系統繪製中文標記時會優先載入 macOS 標準字體（如 STHeiti Medium 或 PingFang），若在 Linux 或 Windows 環境執行，請確保系統已安裝支援中文的字型檔。

    🚀 快速啟動網頁版介面 (Web UI)

    請在終端機（Terminal）中輸入以下指令，即可直接開啟網頁版上傳圖片進行檢測：
    Bash

    python3 ok_ng.py --web

    執行後，系統將會自動開啟瀏覽器（預設為 http://localhost:5000），即可體驗拖曳上傳與即時 OK/NG 判定功能！

🚀 更多使用說明
1. 資料夾批次檢測 (CLI 模式) 📂

檢測指定資料夾內的所有圖片（支援 .jpg, .jpeg, .png），並將結果與標記圖片輸出至指定目錄：
Bash

python3 ok_ng.py /path/to/image_folder --out ./results

2. 單張圖片檢測 🖼️
Bash

python3 ok_ng.py /path/to/single_image.png --out ./results

3. 互動式輸入 💬

若執行時未帶入任何參數，程式將會主動提示您輸入或拖曳路徑：
Bash

python3 ok_ng.py

⚙️ 檢測參數與門檻設定

您可以在程式碼開頭微調以下參數以適應不同的相機距離與光線：
Python

# 接腳檢測興趣區域 (ROI: [x_start, y_start, x_end, y_end])
LEFT_PRONG_ROI = [1120, 800, 1200, 880]
RIGHT_PRONG_ROI = [1300, 800, 1400, 880]

# 判斷門檻
DARK_PIXEL_THRESHOLD = 100  # 灰階閥值
LENGTH_THRESHOLD = 80       # 合格長度門檻 (像素 px)

📊 檢測結果範例輸出

命令行批次掃描時的輸出格式如下：
Plaintext

Scanning directory: ./test_images (5 files found)
--------------------------------------------------------------------------------
Filename                                      | Left Length | Right Length | Status
--------------------------------------------------------------------------------
capture_001.jpg                               | 85 px       | 86 px        | OK    
capture_002.jpg                               | 62 px       | 84 px        | NG    
--------------------------------------------------------------------------------
Scan complete. Total: 5, OK: 4, NG: 1
Visualizations saved to: /path/to/results
