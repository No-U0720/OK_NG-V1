import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ok_ng import check_image, AREA_THRESHOLD, MIN_LENGTH_THRESHOLD, MAX_LENGTH_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="接腳檢測回測與閾值驗證工具")
    parser.add_argument("dir_path", nargs="?", default=None, help="包含待檢測圖片的資料夾路徑")
    parser.add_argument("--out", "-o", default="./test_results", help="圖表與輸出資料夾")
    args = parser.parse_args()
    
    dir_path = args.dir_path
    if not dir_path:
        try:
            dir_path = input("請輸入或拖曳包含圖片的資料夾路徑：").strip()
            if (dir_path.startswith('"') and dir_path.endswith('"')) or (dir_path.startswith("'") and dir_path.endswith("'")):
                dir_path = dir_path[1:-1]
        except (KeyboardInterrupt, EOFError):
            print("\n已取消。")
            return

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        print(f"錯誤：路徑 '{dir_path}' 不存在或不是資料夾。")
        return

    # 取得資料夾中所有圖片 (支援遞迴搜尋子資料夾)
    image_extensions = ('.jpg', '.jpeg', '.png')
    files = []
    for root, _, filenames_in_dir in os.walk(dir_path):
        for f in filenames_in_dir:
            if f.lower().endswith(image_extensions):
                files.append(os.path.join(root, f))
    
    if not files:
        print(f"在資料夾或其子資料夾中找不到支援的圖片檔：{dir_path}")
        return

    # 清空舊的輸出資料夾
    if args.out and os.path.exists(args.out):
        import shutil
        print(f"清空舊的回測輸出資料夾: {args.out}")
        try:
            shutil.rmtree(args.out)
        except Exception as e:
            print(f"清空資料夾時發生錯誤: {e}")

    print(f"開始回測資料夾: {dir_path}，共 {len(files)} 張圖片...")
    print("-" * 80)
    
    left_areas = []
    right_areas = []
    left_lens = []
    right_lens = []
    statuses = []
    filenames = []
    
    ok_count = 0
    ng_count = 0

    for f in sorted(files):
        res = check_image(f, os.path.basename(f), output_dir=os.path.join(args.out, "backtest_visuals"))
        if res:
            filenames.append(os.path.basename(f))
            left_areas.append(res["left_area"])
            right_areas.append(res["right_area"])
            left_lens.append(res["left_len"])
            right_lens.append(res["right_len"])
            statuses.append(res["status"])
            
            if res["status"] == "OK":
                ok_count += 1
            else:
                ng_count += 1
            
            print(f"檔案: {os.path.basename(f):<45} | 左面積: {res['left_area']:<6} px, 長度: {res['left_len']:<4} px | 右面積: {res['right_area']:<6} px, 長度: {res['right_len']:<4} px | 狀態: {res['status']}")

    print("-" * 80)
    print(f"回測完成！總數: {len(files)} | OK: {ok_count} | NG: {ng_count}")
    
    # 轉換成 numpy array 以進行統計與圖表繪製
    left_areas = np.array(left_areas)
    right_areas = np.array(right_areas)
    left_lens = np.array(left_lens)
    right_lens = np.array(right_lens)
    statuses = np.array(statuses)
    
    # 建立輸出目錄
    os.makedirs(args.out, exist_ok=True)
    
    # 繪製分佈圖
    plt.figure(figsize=(12, 5))
    
    # 子圖 1：面積分佈散佈圖 (Scatter Plot of Areas)
    plt.subplot(1, 2, 1)
    for status, color, marker in [("OK", "green", "o"), ("NG", "red", "x")]:
        mask = statuses == status
        if np.any(mask):
            plt.scatter(left_areas[mask], right_areas[mask], c=color, label=f"Status: {status}", marker=marker, alpha=0.8, s=60)
            
    # 畫出面積閾值線
    plt.axvline(x=AREA_THRESHOLD, color='blue', linestyle='--', alpha=0.5, label=f'Threshold ({AREA_THRESHOLD})')
    plt.axhline(y=AREA_THRESHOLD, color='blue', linestyle='--', alpha=0.5)
    
    plt.title("Prong Area Distribution (OK vs NG)")
    plt.xlabel("Left Prong Area (pixels)")
    plt.ylabel("Right Prong Area (pixels)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    # 子圖 2：長度分佈散佈圖 (Scatter Plot of Lengths)
    plt.subplot(1, 2, 2)
    for status, color, marker in [("OK", "green", "o"), ("NG", "red", "x")]:
        mask = statuses == status
        if np.any(mask):
            plt.scatter(left_lens[mask], right_lens[mask], c=color, label=f"Status: {status}", marker=marker, alpha=0.8, s=60)
            
    # 畫出長度閾值線 (上下限)
    plt.axvline(x=MIN_LENGTH_THRESHOLD, color='blue', linestyle='--', alpha=0.5, label=f'Min Thresh ({MIN_LENGTH_THRESHOLD})')
    plt.axvline(x=MAX_LENGTH_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'Max Thresh ({MAX_LENGTH_THRESHOLD})')
    plt.axhline(y=MIN_LENGTH_THRESHOLD, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=MAX_LENGTH_THRESHOLD, color='red', linestyle='--', alpha=0.5)
    
    plt.title("Prong Max Length (Height) Distribution")
    plt.xlabel("Left Prong Length (pixels)")
    plt.ylabel("Right Prong Length (pixels)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    chart_path = os.path.join(args.out, "backtest_distribution_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"\n[驗證報告]")
    print(f"1. 數據統計分佈圖已儲存至: {os.path.abspath(chart_path)}")
    print(f"2. 標記長度後的新檢測影像已輸出至: {os.path.abspath(os.path.join(args.out, 'backtest_visuals'))}")
    
if __name__ == "__main__":
    main()
