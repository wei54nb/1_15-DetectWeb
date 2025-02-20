# Fitness Exercise Detection Backend

這是一個基於 Flask、YOLOv8 和 MediaPipe 的後端服務，主要用於運動姿態檢測與計數，支持即時檢測與影片分析。該專案使用 MySQL 資料庫儲存使用者資訊與運動記錄，並提供 API 以支援前端應用。

## 主要功能
- **運動姿態檢測**：使用 YOLOv8 與 MediaPipe 進行姿態估計
- **支援多種運動**：包含深蹲 (squat)、二頭肌彎舉 (bicep curl)、肩上推舉 (shoulder press)、引體向上 (pull-up)、伏地挺身 (push-up) 等
- **即時視訊處理**：提供實時影像串流與姿態檢測
- **自動計數**：根據姿態角度變化自動計算運動次數
- **影片上傳與處理**：支援影片上傳，並輸出標註後的影片
- **MySQL 整合**：使用 MySQL 存儲使用者與運動記錄
- **API 端點**：提供前端使用的 RESTful API

## 環境與套件需求

### 軟體需求
- **Python 3.8** 或以上（本專案測試使用 Python 3.8.20）
- **MySQL Server**：請先啟動 MySQL 服務，並建立資料庫 `nkust_exercise`
- **NVIDIA GPU (選用)**：使用 NVIDIA GPU 可加速模型推論

### 硬體需求
- 建議配備 NVIDIA GPU 以獲得更佳的運算效能，否則可使用 CPU 模式運行。

## 環境設置

### 1. 建立虛擬環境 (推薦)
```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 2. 安裝依賴套件
```sh
pip install -r requirements.txt
```

### 3. Requirements File (requirements.txt)
```txt
Flask==2.2.5
Flask-SocketIO==5.5.1
Flask-Login==0.6.3
Flask-Bcrypt==1.0.1
mysql-connector-python==9.0.0
PyMySQL==1.1.1
pandas==1.5.3
opencv-python==4.10.0
numpy==1.23.5
ultralytics==8.0.55
torch==2.1.1+cu118
mediapipe==0.10.11
eventlet==0.38.2
Werkzeug==2.2.3
XlsxWriter==3.2.2
```

## GPU 與 CUDA / cuDNN 支援

本專案在 PyTorch **2.1.1+cu118** 環境下測試，該版本內建 CUDA 11.8 支援。

- **CUDA 版本**：11.8
- **cuDNN 版本**：建議使用 cuDNN v8.6 或更新版本。
- **NVIDIA 驅動**：565.90（支援 CUDA 12.7，但向下相容 CUDA 11.8）。

## 專案結構
```
.
├── app.py                 # 主後端程式
├── requirements.txt       # 依賴套件清單
├── static/                # 靜態檔案 (如模型權重)
├── templates/             # HTML 模板
├── uploads/               # 上傳的影片存放目錄
├── output/                # 處理後影片存放目錄
└── README.md              # 專案說明文件
```

## 伺服器啟動

### 1. 設定 MySQL 資料庫
確認 MySQL 服務已啟動，並根據需求更新 `app.py` 中的資料庫連線設定（`db_config`）。

### 2. 啟動 Flask 伺服器
```sh
python app.py
```

## API 端點說明

### 1. 上傳影片與姿態分析
- **Endpoint**: `POST /upload`
- **參數**: `file`（影片檔案）、`exercise`（運動類型）
- **回應**: 返回處理後的影片與檢測資訊

### 2. 啟動即時偵測
- **Endpoint**: `POST /start_detection`
- **參數**: `exercise_type`（運動類型）
- **回應**: 返回是否成功啟動即時偵測

### 3. 停止即時偵測
- **Endpoint**: `POST /stop_detection`
- **回應**: 返回是否成功停止即時偵測

### 4. 匯出運動記錄 (Excel)
- **Endpoint**: `GET /export_excel`
- **回應**: 返回包含運動數據的 Excel 檔案

## 其他設定

### GPU 緩存管理
在影片處理與模型推論過程中，會呼叫 `torch.cuda.empty_cache()` 以釋放 GPU 記憶體，確保運行效率。

## Contributing
歡迎提出 Issue 或提交 Pull Request 以改進專案。

## License
本專案採用 MIT License 授權。

## 聯絡方式
如有任何問題，請開啟 GitHub Issue 或聯絡專案維護者。

**Author: 鄭磬蔚**

