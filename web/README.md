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


## GPU 與 CUDA / cuDNN 支援

本專案在 PyTorch **2.1.1+cu118** 環境下測試，該版本內建 CUDA 11.8 支援。
儘管從 `nvidia-smi` 你可以看到系統支援 CUDA 12.7，但 PyTorch 是以 CUDA 11.8 編譯，故請確認下列項目：

- **PyTorch 版本**：2.1.1+cu118  
  執行以下程式確認版本：
  ```python
  import torch
  print(torch.__version__)  # 應輸出 2.1.1+cu118
  ```

- **CUDA 版本**：11.8  
  建議安裝與 CUDA 11.8 相容的 CUDA 工具包和 cuDNN。
  你可以用下列指令確認：
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.version.cuda)  # 預期輸出：11.8
  ```

- **cuDNN 版本**：請使用與 CUDA 11.8 相容的 cuDNN（例如 cuDNN v8.6 或更新版本）。

- **NVIDIA 驅動**：根據 `nvidia-smi`，你的驅動版本為 **565.90**，支援 CUDA 12.7，但同時向下相容於 CUDA 11.8。

在 `app.py` 中，GPU 的配置如下：
```python
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("Using CPU")
```




## MySQL 資料庫設定

請先建立資料庫 `nkust_exercise`，接著依照以下 SQL 指令建立必要的資料表：

### 建立 `courses` 表
```sql
CREATE TABLE courses (
  course_id INT AUTO_INCREMENT PRIMARY KEY,
  course_name VARCHAR(100) NOT NULL,
  teacher_id VARCHAR(50) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 建立 `discussions` 表
```sql
CREATE TABLE discussions (
  discussion_id INT AUTO_INCREMENT PRIMARY KEY,
  course_id INT NOT NULL,
  student_id INT,
  title VARCHAR(200) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  teacher_id INT
);
```

### 建立 `exercise_info` 表
```sql
CREATE TABLE exercise_info (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_id VARCHAR(50) NOT NULL,
  weight DECIMAL(5,2) NOT NULL,
  reps INT NOT NULL,
  sets INT NOT NULL,
  exercise_type VARCHAR(50) NOT NULL,
  timestamp DATETIME NOT NULL
);
```

### 建立 `exercise_records` 表
```sql
CREATE TABLE exercise_records (
  id INT AUTO_INCREMENT PRIMARY KEY,
  monster_count INT NOT NULL DEFAULT 0,
  exercise_type VARCHAR(50) NOT NULL,
  timestamp DATETIME NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 建立 `responses` 表
```sql
CREATE TABLE responses (
  response_id INT AUTO_INCREMENT PRIMARY KEY,
  discussion_id INT NOT NULL,
  user_id VARCHAR(50) NOT NULL,
  content TEXT NOT NULL,
  is_teacher TINYINT(1) DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 建立 `users` 表
```sql
CREATE TABLE users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role ENUM('student', 'teacher') NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 伺服器啟動

### 1. 設定 MySQL 資料庫
確認 MySQL 服務已啟動，並根據需求更新 `app.py` 中的資料庫連線設定（`db_config`）。

### 2. 啟動 Flask 伺服器
```sh
python app.py
```

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

