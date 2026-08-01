import os
import argparse
import numpy as np
import io
import json
import webbrowser
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

# Regions of interest (ROIs) for left and right prongs
# Coordinates format: [x_start, y_start, x_end, y_end]
LEFT_PRONG_ROI = [1120, 800, 1200, 880]
RIGHT_PRONG_ROI = [1300, 800, 1400, 880]

# Thresholds
DARK_PIXEL_THRESHOLD = 100
AREA_THRESHOLD = 1250
LENGTH_THRESHOLD = 80

def check_image(image_path_or_bytes, filename="uploaded_image.png", output_dir=None):
    """
    Check if the image is OK or NG based on prong length (pixel area).
    If output_dir is provided, saves a visualization.
    Can accept a file path or raw bytes.
    """
    try:
        if isinstance(image_path_or_bytes, bytes):
            nparr = np.frombuffer(image_path_or_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_path_or_bytes)
            
        if img is None:
            raise ValueError("Failed to load image.")
            
        h, w = img.shape[:2]
        scale = w / 2592.0
        
        # Convert BGR to RGB for rembg AI background removal
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgba_rgb = remove(img_rgb)
        vis_img = cv2.cvtColor(rgba_rgb, cv2.COLOR_RGBA2BGRA)
        
        # Extract alpha channel as mask
        alpha = vis_img[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
        # Convert to grayscale for thresholding/area measurements
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Keep all pixels below DARK_PIXEL_THRESHOLD (100) to ensure prongs are not cut off
        mask[gray < DARK_PIXEL_THRESHOLD] = 255
        
        # Apply background transparent color (white transparent: 255, 255, 255, 0)
        vis_img[mask == 0] = (255, 255, 255, 0)
        
        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the green contours
        if contours:
            for c in contours:
                if cv2.contourArea(c) > 100 * scale:
                    cv2.drawContours(vis_img, [c], -1, (0, 255, 0, 255), 3)
            
        # Scale ROIs according to image width
        # (scale is already defined above)
        
        # Crop ROIs from original gray image and calculate dark pixel area for backward compatibility
        left_roi = gray[LEFT_PRONG_ROI[1]:LEFT_PRONG_ROI[3], LEFT_PRONG_ROI[0]:LEFT_PRONG_ROI[2]]
        right_roi = gray[RIGHT_PRONG_ROI[1]:RIGHT_PRONG_ROI[3], RIGHT_PRONG_ROI[0]:RIGHT_PRONG_ROI[2]]
        
        left_area = int(np.sum(left_roi < DARK_PIXEL_THRESHOLD))
        right_area = int(np.sum(right_roi < DARK_PIXEL_THRESHOLD))
        
        left_len = 0
        right_len = 0
        
        geom_details = []
        
        # Detect tip and base corners geometrically
        for name, roi_coords, idx in [("Left", LEFT_PRONG_ROI, 1), ("Right", RIGHT_PRONG_ROI, 2)]:
            rx1 = int((roi_coords[0] - 30) * scale)
            ry1 = int((roi_coords[1] - 80) * scale)
            rx2 = int((roi_coords[2] + 30) * scale)
            ry2 = int((roi_coords[3] + 30) * scale)
            
            # Find contour points in this ROI from all significant contours
            pts_in_roi = []
            if contours:
                for c in contours:
                    if cv2.contourArea(c) > 100 * scale:
                        for pt in c:
                            x, y = pt[0]
                            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                                pts_in_roi.append((x, y))
                        
            if len(pts_in_roi) > 5:
                # Tip point (middle of the bottom-most points)
                max_y = max(p[1] for p in pts_in_roi)
                bottom_pts = [p for p in pts_in_roi if p[1] >= max_y - 2 * scale]
                tip_x = int(round(np.mean([p[0] for p in bottom_pts])))
                tip_y = max_y
                tip_pt = (tip_x, tip_y)
                
                # Separate points into left and right of tip to find base corners
                left_pts = [p for p in pts_in_roi if p[0] < tip_pt[0] - 5 * scale]
                right_pts = [p for p in pts_in_roi if p[0] > tip_pt[0] + 5 * scale]
                
                if left_pts and right_pts:
                    base_l = min(left_pts, key=lambda p: p[1])
                    base_r = min(right_pts, key=lambda p: p[1])
                    
                    x1, y1 = base_l
                    x2, y2 = base_r
                    xt, yt = tip_pt
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    line_len = np.hypot(dx, dy)
                    
                    if line_len > 0:
                        dist = abs(dy * xt - dx * yt + x2 * y1 - y2 * x1) / line_len
                        
                        # Projection of tip onto the base line
                        t = ((xt - x1) * dx + (yt - y1) * dy) / (line_len ** 2)
                        proj_x = int(x1 + t * dx)
                        proj_y = int(y1 + t * dy)
                        
                        if name == "Left":
                            left_len = int(round(dist))
                        else:
                            right_len = int(round(dist))
                            
                        # Draw geometry elements on vis_img (BGRA)
                        cv2.line(vis_img, base_l, base_r, (255, 0, 0, 255), 3) # Blue base line connecting yellow points
                        cv2.line(vis_img, tip_pt, (proj_x, proj_y), (255, 0, 255, 255), 3) # Magenta perpendicular line
                        cv2.circle(vis_img, base_l, 6, (0, 255, 255, 255), -1) # Yellow left base point
                        cv2.circle(vis_img, base_r, 6, (0, 255, 255, 255), -1) # Yellow right base point
                        cv2.circle(vis_img, tip_pt, 6, (0, 0, 255, 255), -1) # Red tip point
                        geom_details.append({
                            "tip_num": idx,
                            "xt": xt, "yt": yt,
                            "proj_x": proj_x, "proj_y": proj_y,
                            "dist": dist
                        })
                        
        left_ok = left_len >= LENGTH_THRESHOLD
        right_ok = right_len >= LENGTH_THRESHOLD
        
        is_ok = left_ok and right_ok
        status_str = "OK" if is_ok else "NG"
        
        result = {
            "status": status_str,
            "left_area": left_area,
            "right_area": right_area,
            "left_len": left_len,
            "right_len": right_len,
            "left_ok": left_ok,
            "right_ok": right_ok
        }
        
        # Load custom font for beautiful rendering
        try:
            # STHeiti Medium is standard and verified on macOS
            font_title = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 70)
            font_body = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 45)
            font_label = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 24)
        except Exception:
            try:
                # Fallback to PingFang
                font_title = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 70)
                font_body = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 45)
                font_label = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
            except Exception:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
                font_label = ImageFont.load_default()
            
        # Convert BGRA to RGBA to draw text using PIL
        vis_img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGRA2RGBA))
        draw = ImageDraw.Draw(vis_img_pil)
        
        # Draw status title
        status_color = (0, 220, 0) if is_ok else (255, 50, 50)
        draw.text((80, 80), f"檢測結果：{status_str}", fill=status_color, font=font_title)
        
        # Compile explanations
        reasons = []
        if is_ok:
            reasons.append(f"兩側接腳均符合標準。左接腳長度: {left_len} px，右接腳長度: {right_len} px (標準: >= {LENGTH_THRESHOLD} px)")
        else:
            if not left_ok:
                reasons.append(f"左側接腳過短 (量測長度: {left_len} px，標準: >= {LENGTH_THRESHOLD} px)")
            if not right_ok:
                reasons.append(f"右側接腳過短 (量測長度: {right_len} px，標準: >= {LENGTH_THRESHOLD} px)")
                
        for idx, reason in enumerate(reasons):
            draw.text((80, 180 + idx * 70), reason, fill=(255, 255, 255), font=font_body)
            
        # Draw geometric text labels
        for detail in geom_details:
            t_num = detail["tip_num"]
            xt, yt = detail["xt"], detail["yt"]
            px, py = detail["proj_x"], detail["proj_y"]
            d = detail["dist"]
            
            # Tip coordinate label in red
            draw.text((xt + 12, yt - 10), f"tip{t_num}", fill=(255, 0, 0), font=font_label)
            draw.text((xt + 12, yt + 15), f"({xt},{yt})", fill=(255, 0, 0), font=font_label)
            
            # Length label in magenta next to the line
            draw.text((px + 12, py - 10), f"L = {d:.1f} px", fill=(255, 0, 255), font=font_label)
            
        # Convert back to BGRA (OpenCV)
        vis_img = cv2.cvtColor(np.array(vis_img_pil), cv2.COLOR_RGBA2BGRA)
        
        # Convert processed image back to bytes or save to disk
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(output_dir, f"result_{base_name}.png")
            cv2.imwrite(output_path, vis_img)
            result["visualized_path"] = output_path
            
        # We also keep the annotated bytes in the result dictionary to return via server (PNG format)
        _, img_encoded = cv2.imencode('.png', vis_img)
        result["image_bytes"] = img_encoded.tobytes()
        
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# HTML / Web UI Template
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VKD Defect Detection System</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --header-bg: #ffffff;
            --panel-bg: #ffffff;
            --text-color: #333333;
            --accent-blue: #1e40af;
            --accent-light-blue: #3b82f6;
            --success-color: #2ca02c;
            --danger-color: #d62728;
            --border-color: #dddddd;
            --dark-terminal-bg: #0c0f1d;
            --teal-highlight: #00bcd4;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Noto Sans TC', sans-serif;
            padding: 10px 20px;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        /* Header Styles */
        .system-header {
            background-color: var(--header-bg);
            border-bottom: 3px solid var(--accent-blue);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo-placeholder {
            font-weight: 900;
            font-size: 24px;
            color: #0f172a;
            letter-spacing: -1px;
            display: flex;
            align-items: center;
        }

        .logo-placeholder span {
            color: #ef4444;
            font-size: 14px;
            font-weight: 700;
            border: 1.5px solid #ef4444;
            padding: 1px 3px;
            margin-left: 5px;
            border-radius: 3px;
        }

        .header-title {
            font-size: 26px;
            font-weight: 800;
            color: var(--accent-blue);
            text-align: center;
            flex-grow: 1;
        }

        .header-right {
            font-size: 14px;
            color: #666666;
            font-weight: 700;
        }

        /* Dashboard Layout */
        .main-dashboard {
            display: grid;
            grid-template-columns: 1.1fr 1.1fr 0.8fr;
            gap: 15px;
            flex-grow: 1;
        }

        .column-panel {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }

        .panel-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #1e293b;
        }

        /* Left Column: Live stream */
        .live-feed-box {
            background-color: #000;
            border: 3px solid #10b981;
            border-radius: 6px;
            position: relative;
            aspect-ratio: 4/3;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .live-meta-bar {
            background-color: #111827;
            display: flex;
            justify-content: space-around;
            padding: 4px;
            font-size: 11px;
            color: #ffffff;
            font-weight: 700;
            border-bottom: 1px solid #374151;
        }

        .meta-item span {
            color: var(--teal-highlight);
        }

        .live-image-container {
            position: relative;
            flex-grow: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #1f2937;
        }

        .live-image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .live-overlay-feed {
            position: absolute;
            top: 8px;
            left: 8px;
            background-color: rgba(0,0,0,0.6);
            color: #ffffff;
            padding: 2px 6px;
            font-size: 9px;
            font-weight: 700;
            border-radius: 3px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .live-dot {
            width: 6px;
            height: 6px;
            background-color: #ff4d4d;
            border-radius: 50%;
            display: inline-block;
            animation: blink 1s infinite;
        }

        .live-overlay-device {
            position: absolute;
            top: 8px;
            right: 8px;
            background-color: rgba(0,0,0,0.6);
            color: #cccccc;
            padding: 2px 6px;
            font-size: 9px;
        }

        /* OK/NG Badges */
        .result-overlay-badge {
            position: absolute;
            top: 15%;
            right: 5%;
            padding: 5px 15px;
            font-weight: 900;
            font-size: 24px;
            border-radius: 4px;
            border: 2px solid #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            z-index: 10;
        }

        .badge-ok {
            background-color: var(--success-color);
            color: white;
        }

        .badge-ng {
            background-color: var(--danger-color);
            color: white;
        }

        .live-controls {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
            padding: 0 5px;
        }

        .toggle-mode {
            font-size: 14px;
            font-weight: 700;
            color: #d62728;
        }

        .toggle-mode span {
            color: #333333;
            margin-left: 10px;
        }

        .btn-capture {
            background-color: #3b82f6;
            color: #ffffff;
            border: none;
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 700;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .btn-capture:hover {
            background-color: #2563eb;
        }

        /* Middle Column: Results */
        .result-view-box {
            background-color: #000;
            border: 3px solid #10b981;
            border-radius: 6px;
            aspect-ratio: 4/3;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #1f2937;
            position: relative;
        }

        .result-view-box img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .result-info-box {
            margin-top: 10px;
            display: flex;
            gap: 15px;
        }

        .result-text-summary {
            font-size: 14px;
            font-weight: 700;
            color: #111827;
            flex-grow: 1;
        }

        .result-status-title {
            font-size: 16px;
            margin-bottom: 5px;
        }

        .config-red-table {
            background-color: #000000;
            color: #ff3b30;
            border: 1px solid #333;
            padding: 8px;
            font-family: monospace;
            font-size: 11px;
            line-height: 1.4;
            min-width: 150px;
        }

        .config-label-arrow {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 700;
            color: #d62728;
        }

        .config-arrow {
            font-size: 20px;
        }

        /* Right Column: History */
        .history-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #e2e8f0;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
        }

        .history-play-btn {
            font-size: 12px;
            cursor: pointer;
            background: none;
            border: none;
            color: #475569;
            font-weight: 700;
        }

        .history-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow-y: auto;
            flex-grow: 1;
            max-height: 380px;
            padding-right: 2px;
        }

        .history-item {
            display: flex;
            border: 2px solid #94a3b8;
            border-radius: 4px;
            background-color: #f8fafc;
            cursor: pointer;
            overflow: hidden;
            height: 70px;
            position: relative;
        }

        .history-item.ok-border {
            border-color: var(--success-color);
        }

        .history-item.ng-border {
            border-color: var(--danger-color);
        }

        .history-item-num {
            width: 30px;
            background-color: #e2e8f0;
            display: flex;
            justify-content: center;
            align-items: center;
            font-weight: 700;
            font-size: 12px;
            border-right: 1px solid #cbd5e1;
        }

        .history-item-thumb {
            flex-grow: 1;
            height: 100%;
            background-color: #0f172a;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .history-item-thumb img {
            height: 100%;
            width: 100%;
            object-fit: contain;
        }

        .history-badge {
            position: absolute;
            top: 2px;
            right: 2px;
            font-size: 8px;
            font-weight: 900;
            padding: 1px 4px;
            border-radius: 2px;
            color: #fff;
        }

        /* Bottom Row: Terminal Logs */
        .system-log-section {
            margin-top: 15px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px;
            background-color: #ffffff;
        }

        .terminal-box {
            background-color: var(--dark-terminal-bg);
            color: #38bdf8;
            font-family: monospace;
            font-size: 12px;
            padding: 10px;
            height: 120px;
            overflow-y: auto;
            border-radius: 4px;
            line-height: 1.5;
        }

        .terminal-time {
            color: #a3e635;
        }

        .terminal-post {
            color: #f472b6;
        }

        .terminal-elapsed {
            color: #fb923c;
        }

        /* Animations */
        @keyframes blink {
            0% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
        }

        .empty-placeholder {
            color: #94a3b8;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <!-- Hidden File Input -->
    <input type="file" id="file-input" accept="image/*" style="display: none;">

    <header class="system-header">
        <div class="header-left">
            <div class="logo-placeholder">VKD <span>KING DUAN</span></div>
        </div>
        <div class="header-title">VKD Defect Detection System</div>
        <div class="header-right">Ver 1.0 20260710</div>
    </header>

    <main class="main-dashboard">
        <!-- Column 1: Live stream -->
        <section class="column-panel">
            <h2 class="panel-title">Live stream</h2>
            <div class="live-feed-box">
                <div class="live-meta-bar">
                    <div class="meta-item">客戶名稱: <span>KB</span></div>
                    <div class="meta-item">目前產品: <span>020-001</span></div>
                    <div class="meta-item">良率: <span id="yield-rate">99.9%</span></div>
                    <div class="meta-item">站別: <span>01</span></div>
                </div>
                <div class="live-image-container">
                    <div class="live-overlay-feed"><span class="live-dot"></span> LIVE FEED</div>
                    <div class="live-overlay-device">Device: UVC Camera 0</div>
                    <div class="result-overlay-badge badge-ok" id="live-badge" style="display: none;">OK</div>
                    <img id="live-img" style="display: none;">
                    <div class="empty-placeholder" id="live-placeholder">等待相機影像擷取...</div>
                </div>
            </div>
            <div class="live-controls">
                <button class="btn-capture" id="btn-capture">
                    📤 上傳圖片
                </button>
            </div>
        </section>

        <!-- Column 2: Latest Result Image -->
        <section class="column-panel">
            <h2 class="panel-title">Latest Inspection Result Image</h2>
            <div class="result-view-box">
                <img id="result-img" style="display: none;">
                <div class="empty-placeholder" id="result-placeholder">等待量測影像輸出...</div>
            </div>
            <div class="result-info-box">
                <div class="result-text-summary">
                    <div class="result-status-title" id="result-status-text">請上傳影像進行檢測</div>
                    <div style="font-size: 11px; color: #666;" id="result-timestamp">---</div>
                </div>
                <div class="config-label-arrow">
                    <div class="config-red-table">
                        左側長度: <span id="val-left-len">--</span> px<br>
                        右側長度: <span id="val-right-len">--</span> px<br>
                        設定門檻: 80.00 px<br>
                        量測狀態: <span id="val-status">--</span>
                    </div>
                    <div class="config-arrow">⬅</div>
                    <div style="font-size: 11px;">瑕疵檢測<br>設定值</div>
                </div>
            </div>
        </section>

        <!-- Column 3: History -->
        <section class="column-panel">
            <h2 class="panel-title">History 預設5個</h2>
            <div class="history-controls">
                <button class="history-play-btn">▶ PLAY</button>
                <div style="font-size: 14px; font-weight: 700; color: #475569;">+</div>
            </div>
            <div class="history-list" id="history-list">
                <!-- Prepopulated or dynamically added items -->
                <div class="empty-placeholder" style="margin-top: 50px;">尚無檢測紀錄</div>
            </div>
        </section>
    </main>

    <!-- Bottom Row: System Log -->
    <footer class="system-log-section">
        <h2 class="panel-title" style="font-size: 16px;">系統日誌</h2>
        <div class="terminal-box" id="terminal-box">
            <div>VKD Defect Detection System initialized. Ready for inspection.</div>
        </div>
    </footer>

    <script>
        const btnCapture = document.getElementById('btn-capture');
        const fileInput = document.getElementById('file-input');
        
        const liveImg = document.getElementById('live-img');
        const livePlaceholder = document.getElementById('live-placeholder');
        const liveBadge = document.getElementById('live-badge');
        
        const resultImg = document.getElementById('result-img');
        const resultPlaceholder = document.getElementById('result-placeholder');
        const resultStatusText = document.getElementById('result-status-text');
        const resultTimestamp = document.getElementById('result-timestamp');
        
        const valLeftLen = document.getElementById('val-left-len');
        const valRightLen = document.getElementById('val-right-len');
        const valStatus = document.getElementById('val-status');
        const yieldRateSpan = document.getElementById('yield-rate');
        
        const historyList = document.getElementById('history-list');
        const terminalBox = document.getElementById('terminal-box');

        let historyItems = [];
        let totalCount = 0;
        let okCount = 0;

        btnCapture.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processFile(e.target.files[0]);
            }
        });

        // Add drag & drop support to live feed area
        const liveFeedBox = document.querySelector('.live-feed-box');
        liveFeedBox.addEventListener('dragover', (e) => e.preventDefault());
        liveFeedBox.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFile(files[0]);
            }
        });

        // Write log entry to terminal
        function logToTerminal(message, isRequest = false, elapsed = null) {
            const div = document.createElement('div');
            const now = new Date();
            const timeStr = now.toLocaleTimeString();
            
            if (isRequest) {
                div.innerHTML = `<span class="terminal-time">[${timeStr}]</span> <span class="terminal-post">POST /upload</span> HTTP/1.1 200 OK`;
                if (elapsed !== null) {
                    const elapsedDiv = document.createElement('div');
                    elapsedDiv.innerHTML = `&nbsp;&nbsp;辨識耗時: <span class="terminal-elapsed">${elapsed.toFixed(2)} 秒</span>`;
                    terminalBox.appendChild(div);
                    terminalBox.appendChild(elapsedDiv);
                } else {
                    terminalBox.appendChild(div);
                }
            } else {
                div.innerHTML = `<span class="terminal-time">[${timeStr}]</span> ${message}`;
                terminalBox.appendChild(div);
            }
            terminalBox.scrollTop = terminalBox.scrollHeight;
        }

        async function processFile(file) {
            // Show original image in live stream view
            const reader = new FileReader();
            reader.onload = function(e) {
                liveImg.src = e.target.result;
                liveImg.style.display = 'block';
                livePlaceholder.style.display = 'none';
            }
            reader.readAsDataURL(file);

            logToTerminal(`Uploading and capturing file: ${file.name}...`);

            try {
                const response = await fetch(`/upload?filename=${encodeURIComponent(file.name)}`, {
                    method: 'POST',
                    body: file
                });

                if (!response.ok) {
                    throw new Error('影像處理失敗');
                }

                const result = await response.json();
                const now = new Date();
                const formattedTime = now.getFullYear() + '-' + 
                                      String(now.getMonth()+1).padStart(2, '0') + '-' + 
                                      String(now.getDate()).padStart(2, '0') + ' ' + 
                                      String(now.getHours()).padStart(2, '0') + ':' + 
                                      String(now.getMinutes()).padStart(2, '0');

                // Update Result Image view
                resultImg.src = `data:image/png;base64,${result.image_base64}`;
                resultImg.style.display = 'block';
                resultPlaceholder.style.display = 'none';

                // Update Live view Badge
                liveBadge.innerText = result.status;
                liveBadge.className = `result-overlay-badge ${result.status === 'OK' ? 'badge-ok' : 'badge-ng'}`;
                liveBadge.style.display = 'block';

                // Update info panels
                resultStatusText.innerText = result.status === 'OK' ? `${formattedTime} Passed` : `${formattedTime} Failed`;
                resultStatusText.style.color = result.status === 'OK' ? 'var(--success-color)' : 'var(--danger-color)';
                resultTimestamp.innerText = `影像名稱: ${file.name}`;

                valLeftLen.innerText = result.left_len;
                valRightLen.innerText = result.right_len;
                valStatus.innerText = result.status === 'OK' ? 'Passed' : 'Failed';
                valStatus.style.color = result.status === 'OK' ? 'var(--success-color)' : 'var(--danger-color)';

                // Update stats & yield rate
                totalCount++;
                if (result.status === 'OK') okCount++;
                const yieldRate = ((okCount / totalCount) * 100).toFixed(1);
                yieldRateSpan.innerText = `${yieldRate}%`;

                // Log to terminal
                logToTerminal(null, true, result.elapsed);
                logToTerminal(`檢測結果: ${result.status} (左接腳: ${result.left_len} px, 右接腳: ${result.right_len} px)`);

                // Append to history list
                addHistoryItem(file.name, liveImg.src, resultImg.src, result);

            } catch (error) {
                logToTerminal(`錯誤: ${error.message}`);
                alert('處理失敗：' + error.message);
            }
        }

        function addHistoryItem(name, origSrc, resultSrc, result) {
            // Keep maximum 5 items in list
            if (historyItems.length >= 5) {
                historyItems.pop();
            }

            const item = {
                id: ++totalCount,
                name: name,
                origSrc: origSrc,
                resultSrc: resultSrc,
                result: result,
                time: new Date().toLocaleTimeString()
            };

            historyItems.unshift(item);
            renderHistory();
        }

        function renderHistory() {
            historyList.innerHTML = '';
            if (historyItems.length === 0) {
                historyList.innerHTML = `<div class="empty-placeholder" style="margin-top: 50px;">尚無檢測紀錄</div>`;
                return;
            }

            historyItems.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = `history-item ${item.result.status === 'OK' ? 'ok-border' : 'ng-border'}`;
                div.innerHTML = `
                    <div class="history-item-num">${item.id}</div>
                    <div class="history-item-thumb">
                        <img src="${item.resultSrc}">
                    </div>
                    <div class="history-badge ${item.result.status === 'OK' ? 'badge-ok' : 'badge-ng'}">${item.result.status}</div>
                `;
                
                // Clicking history item restores it on panels
                div.addEventListener('click', () => {
                    liveImg.src = item.origSrc;
                    liveImg.style.display = 'block';
                    livePlaceholder.style.display = 'none';

                    liveBadge.innerText = item.result.status;
                    liveBadge.className = `result-overlay-badge ${item.result.status === 'OK' ? 'badge-ok' : 'badge-ng'}`;
                    liveBadge.style.display = 'block';

                    resultImg.src = item.resultSrc;
                    resultImg.style.display = 'block';
                    resultPlaceholder.style.display = 'none';

                    resultStatusText.innerText = item.result.status === 'OK' ? `History Passed` : `History Failed`;
                    resultStatusText.style.color = item.result.status === 'OK' ? 'var(--success-color)' : 'var(--danger-color)';
                    resultTimestamp.innerText = `影像名稱: ${item.name} (${item.time})`;

                    valLeftLen.innerText = item.result.left_len;
                    valRightLen.innerText = item.result.right_len;
                    valStatus.innerText = item.result.status === 'OK' ? 'Passed' : 'Failed';
                    valStatus.style.color = item.result.status === 'OK' ? 'var(--success-color)' : 'var(--danger-color)';
                });

                historyList.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

# Web server implementation using built-in http.server
class WebUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default terminal logs
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/upload"):
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                return

            # Read raw image bytes
            image_bytes = self.rfile.read(content_length)
            
            import time
            import base64
            start_time = time.time()
            res = check_image(image_bytes, filename="web_upload.png")
            elapsed = time.time() - start_time
            
            if res:
                # Add base64 representation of the processed image to return in JSON
                base64_image = base64.b64encode(res["image_bytes"]).decode('utf-8')
                
                response_data = {
                    "status": res["status"],
                    "left_area": res["left_area"],
                    "right_area": res["right_area"],
                    "left_len": res["left_len"],
                    "right_len": res["right_len"],
                    "left_ok": res["left_ok"],
                    "right_ok": res["right_ok"],
                    "image_base64": base64_image,
                    "elapsed": elapsed
                }
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run_web_server(port=5001):
    import socket
    local_ip = "localhost"
    try:
        # Get actual local IP address on the network
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
        
    server = HTTPServer(('0.0.0.0', port), WebUIHandler)
    url_local = f"http://localhost:{port}"
    url_network = f"http://{local_ip}:{port}"
    
    print(f"正在啟動本機網頁伺服器（已開放所有網路介面）...")
    print(f"本機存取網址：{url_local}")
    print(f"同網域（其他人）存取網址：{url_network}")
    print("按下 Ctrl+C 可停止伺服器。")
    webbrowser.open(url_local)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n伺服器已停止。")

def main():
    parser = argparse.ArgumentParser(description="Prong Length Detection Tool (OK/NG)")
    parser.add_argument("path", nargs="?", default=None, help="Path to an image file or directory containing images")
    parser.add_argument("--out", "-o", default="./results", help="Output directory to save visual results")
    parser.add_argument("--web", action="store_true", help="Launch interactive web browser upload UI")
    args = parser.parse_args()
    
    if args.web:
        run_web_server()
        return
        
    path = args.path
    if not path:
        # Prompt user interactively if no path is provided
        try:
            path = input("請拖曳圖片/資料夾到這裡，或輸入路徑：").strip()
            # Remove quotes if user dragged and dropped path containing spaces
            if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
                path = path[1:-1]
        except (KeyboardInterrupt, EOFError):
            print("\n已取消。")
            return
            
    if not path:
        print("錯誤：未提供有效的路徑。")
        return
        
    # 清空舊的輸出資料夾
    if args.out and os.path.exists(args.out):
        import shutil
        print(f"清空舊的輸出資料夾: {args.out}")
        try:
            shutil.rmtree(args.out)
        except Exception as e:
            print(f"清空資料夾時發生錯誤: {e}")
            
    if os.path.isdir(path):
        # Process all jpg images in directory
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"No images found in directory: {path}")
            return
        
        print(f"Scanning directory: {path} ({len(files)} files found)")
        print("-" * 80)
        print(f"{'Filename':<45} | {'Left Length':<11} | {'Right Length':<12} | {'Status':<6}")
        print("-" * 80)
        
        ok_count = 0
        ng_count = 0
        
        for f in sorted(files):
            res = check_image(f, os.path.basename(f), args.out)
            if res:
                if res["status"] == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
                left_len_str = f"{res['left_len']} px"
                right_len_str = f"{res['right_len']} px"
                print(f"{os.path.basename(f):<45} | {left_len_str:<11} | {right_len_str:<12} | {res['status']:<6}")
        
        print("-" * 80)
        print(f"Scan complete. Total: {len(files)}, OK: {ok_count}, NG: {ng_count}")
        print(f"Visualizations saved to: {os.path.abspath(args.out)}")
        
    elif os.path.isfile(path):
        print(f"Processing single file: {path}")
        res = check_image(path, os.path.basename(path), args.out)
        if res:
            print(f"Status: {res['status']}")
            print(f"Left Prong Area: {res['left_area']} ({'OK' if res['left_ok'] else 'NG'})")
            print(f"Right Prong Area: {res['right_area']} ({'OK' if res['right_ok'] else 'NG'})")
            if "visualized_path" in res:
                print(f"Visualization saved to: {res['visualized_path']}")
    else:
        print(f"Error: Path '{path}' does not exist.")

if __name__ == "__main__":
    main()
