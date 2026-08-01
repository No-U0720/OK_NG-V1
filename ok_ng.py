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
    <title>接腳長度自動檢測系統 (OK/NG)</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+TC:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 26, 46, 0.6);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-color: #3b82f6;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --text-color: #f3f4f6;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', 'Noto Sans TC', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            overflow-x: hidden;
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeInDown 0.6s ease-out;
        }
        header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        header p {
            color: #9ca3af;
            font-size: 1rem;
        }
        .container {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            animation: fadeInUp 0.8s ease-out;
        }
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(59, 130, 246, 0.1);
        }
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.02);
            flex-grow: 1;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-color);
            background: rgba(59, 130, 246, 0.05);
        }
        .dropzone svg {
            width: 64px;
            height: 64px;
            fill: #9ca3af;
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
        }
        .dropzone:hover svg {
            transform: scale(1.1);
            fill: var(--accent-color);
        }
        .dropzone p {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        .dropzone span {
            color: #6b7280;
            font-size: 0.9rem;
        }
        #file-input {
            display: none;
        }
        .result-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 400px;
            border-radius: 16px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }
        #result-img {
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
            display: none;
        }
        .placeholder-text {
            color: #6b7280;
            font-size: 1.1rem;
            text-align: center;
        }
        .status-badge {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1.5rem;
            border-radius: 30px;
            font-weight: 800;
            font-size: 1.5rem;
            text-transform: uppercase;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            display: none;
        }
        .status-badge.ok {
            background-color: var(--success-color);
            color: white;
        }
        .status-badge.ng {
            background-color: var(--danger-color);
            color: white;
        }
        .stats {
            margin-top: 1.5rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            width: 100%;
            display: none;
        }
        .stat-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .stat-card h3 {
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 0.5rem;
        }
        .stat-card p {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .loader {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 4px solid var(--accent-color);
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <header>
        <h1>接腳長度自動檢測系統</h1>
        <p>上傳或拖曳圖片，即時判定 OK / NG 並畫記量測原因</p>
    </header>

    <div class="container">
        <div class="card">
            <div class="dropzone" id="dropzone">
                <svg viewBox="0 0 24 24">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
                <p>點擊上傳 或 拖曳圖片至此處</p>
                <span>支援格式: JPG, JPEG, PNG</span>
                <input type="file" id="file-input" accept="image/*">
            </div>
        </div>

        <div class="card">
            <div class="result-panel">
                <div class="loader" id="loader"></div>
                <div class="placeholder-text" id="placeholder">等待上傳圖片...</div>
                <img id="result-img" alt="偵測結果影像">
                <div class="status-badge" id="badge"></div>
            </div>
            
            <div class="stats" id="stats">
                <div class="stat-card">
                    <h3>左接腳量測面積</h3>
                    <p id="left-area-val">-</p>
                </div>
                <div class="stat-card">
                    <h3>右接腳量測面積</h3>
                    <p id="right-area-val">-</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('file-input');
        const loader = document.getElementById('loader');
        const placeholder = document.getElementById('placeholder');
        const resultImg = document.getElementById('result-img');
        const badge = document.getElementById('badge');
        const stats = document.getElementById('stats');
        const leftAreaVal = document.getElementById('left-area-val');
        const rightAreaVal = document.getElementById('right-area-val');

        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processFile(e.target.files[0]);
            }
        });

        async function processFile(file) {
            // Show loader
            placeholder.style.display = 'none';
            resultImg.style.display = 'none';
            badge.style.display = 'none';
            stats.style.display = 'none';
            loader.style.display = 'block';

            try {
                // Send raw binary to avoid complex multipart parser on python side
                const response = await fetch(`/upload?filename=${encodeURIComponent(file.name)}`, {
                    method: 'POST',
                    body: file
                });

                if (!response.ok) {
                    throw new Error('影像處理失敗');
                }

                const result = await response.json();
                
                // Set result image (source is base64 formatted png)
                resultImg.src = `data:image/png;base64,${result.image_base64}`;
                resultImg.style.display = 'block';

                // Set badge
                badge.innerText = result.status;
                badge.className = `status-badge ${result.status.toLowerCase()}`;
                badge.style.display = 'block';

                // Set stats
                leftAreaVal.innerText = `${result.left_area} px`;
                leftAreaVal.style.color = result.left_ok ? 'var(--success-color)' : 'var(--danger-color)';
                rightAreaVal.innerText = `${result.right_area} px`;
                rightAreaVal.style.color = result.right_ok ? 'var(--success-color)' : 'var(--danger-color)';
                stats.style.display = 'grid';

            } catch (error) {
                alert('處理失敗：' + error.message);
                placeholder.style.display = 'block';
                placeholder.innerText = '發生錯誤，請重新上傳';
            } finally {
                loader.style.display = 'none';
            }
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
            
            # Check image
            import base64
            res = check_image(image_bytes, filename="web_upload.png")
            
            if res:
                # Add base64 representation of the processed image to return in JSON
                base64_image = base64.b64encode(res["image_bytes"]).decode('utf-8')
                
                response_data = {
                    "status": res["status"],
                    "left_area": res["left_area"],
                    "right_area": res["right_area"],
                    "left_ok": res["left_ok"],
                    "right_ok": res["right_ok"],
                    "image_base64": base64_image
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

def run_web_server(port=5000):
    server = HTTPServer(('localhost', port), WebUIHandler)
    url = f"http://localhost:{port}"
    print(f"正在啟動本機網頁伺服器...")
    print(f"請在瀏覽器中打開此網址：{url}")
    print("按下 Ctrl+C 可停止伺服器。")
    webbrowser.open(url)
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
