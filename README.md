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

### 1. 複製專案
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
