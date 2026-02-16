# JAXA Earth Soundscape (Local AI)
地図クリック → 座標 → 衛星データ（NDVI/LST/降水） → ローカルで
- 画像（Stable Diffusion）
- 音（MusicGen）
を生成する Streamlit アプリです。

---

## 動作環境（重要）
このアプリはローカルAI生成があるため、PC性能差が体験差になります。

### 推奨（快適）
- Windows 11
- NVIDIA GPU（VRAM 8GB以上推奨 / 6GBでも可）
- RAM 16GB以上（推奨 32GB）
- 空き容量 10GB以上（モデルDLで増えます）
- Python 3.10〜3.12

### GPUなし（CPU）でも動く？
- 動きますが、画像/音の生成はかなり遅くなります（仕様です）

---

## 1) まっさらな状態からの起動手順（Windows11 / PowerShell）

### 1-1. Pythonを入れる
PowerShellで確認：
```powershell
python --version
