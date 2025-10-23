# special-project-ver.1

這個 repository 包含數個用於機器學習與時間序列分析的筆記型檔案與訓練資料範例（notebooks、環境設定與範例腳本）。

主要內容
- Jupyter notebooks: `LSTM.ipynb`, `transformer.ipynb`, `XGBoost.ipynb`, `catBoost.ipynb`, `Linear_regression.ipynb`, `poster.ipynb`。
- 環境與依賴: `environment.yml`、`requirements.txt`。
- 範例/訓練資料（未全部上傳，詳見下方資料說明）

如何使用
1. 建議在虛擬環境（conda 或 venv）中使用。若使用 conda，可建立 environment：

   conda env create -f environment.yml
   conda activate <env-name>

2. 啟動 Jupyter Lab/Notebook，然後開啟你要運行的 `.ipynb` 檔案：

   jupyter lab

3. 示範腳本或短程式碼可直接在 `tmp.py` 中執行（視需要修改）。

關於資料 (CSV)
- 本 repo 的 `.gitignore` 目前包含 `*.csv`，因此大多數 CSV 檔案未被追蹤或上傳到 GitHub。若你希望把資料上傳到 repo，請考慮以下幾個選項：
  1. 移除 `.gitignore` 中的 `*.csv`，然後將小於 100MB 的 CSV 直接加入並 push。
  2. 若 CSV 檔案較大或數量很多，建議使用 Git LFS（需在本地安裝 `git-lfs` 並執行 `git lfs track "*.csv"`）。
  3. 或者將資料放在外部儲存（例如 Google Drive、S3），repo 只保留載入資料的程式碼與範例。

授權與貢獻
- 若要我幫你新增一個授權檔（例如 MIT）或 CONTRIBUTING 指南，請回覆我想使用的授權類型，我會自動新增並 push。

聯絡
- 如需其他自動化（例如新增 README 的更多內容、設定 GitHub Actions、或將資料加入 Git LFS），請告訴我你要的選項。
