# 學生端專案檔案說明 — 西洋棋增強式學習競賽（Stable Baselines 3 + MaskablePPO）

本目錄包含參加西洋棋 PvP 競賽所需的所有基礎檔案，使用 **sb3-contrib MaskablePPO** 框架：

## 1. `model.py`（演算法設定，**主要修改點**）
- **用途**：定義 SB3 演算法類別、Policy 與網路結構。
- **設計**：`train.py` 和 `agent.py` 都從這裡 import，**只需改一次即自動同步**。
- **換演算法**：修改 `ALGORITHM = MaskablePPO`（須支援 action masking）。
- **調整網路**：修改 `POLICY_KWARGS` 中的 `net_arch`，例如改成 `[512, 512]`。

## 2. `agent.py`（競賽介面，**必須保留**）
- **用途**：定義系統載入與呼叫你模型的標準介面。
- **規則**：`class Agent` 名稱、`act()` 方法名稱與參數簽名**不可修改**。
- **可改**：將 `deterministic=True` 改為 `False` 以啟用機率採樣。

## 3. `train.py`（訓練腳本，**可自由調整**）
- **用途**：使用 SB3 進行自對弈（Self-Play）訓練，產生 `model.zip`。
- **設計**：`ChessSelfPlayEnv` 將學習 Agent 封裝為 Gymnasium 環境，對手使用隨機合法動作。
- **可改**：
  - `TOTAL_TIMESTEPS`、`N_ENVS` — 訓練規模
  - `_shape_reward()` — 自訂中間步獎勵（如吃子獎勵）
- **執行**：
  ```bash
  python train.py
  ```

## 4. `run.py`（提交腳本）
- **用途**：將 `agent.py`、`model.py`、`model.zip` 上傳至 MLArena 伺服器。
- **自動嵌入**：`model.py` 會自動被嵌入 `agent.py`，沙箱可直接執行。
- **執行前**：修改 `STUDENT_ID`、`SLOT_INDEX`、`SLOT_NAME`、`DESCRIPTION`。
- **執行**：
  ```bash
  python run.py
  ```

## 5. `environment.yml` 與 `requirements.txt`（Conda 環境與套件設定檔）
請確保先使用終端機進入作業包目錄下（即 `student` 資料夾），再執行以下環境建立指令：
```bash
conda env create -f environment.yml
conda activate mlarena
pip install -r requirements.txt
```

## 6. `arena_client.py`（系統客戶端，通常不需修改）

---

## 執行流程

```
1. 開啟終端機，並使用 cd 命令進入解壓後的作業包目錄（即含有 environment.yml 的資料夾）：
   cd <作業包目錄路徑>

   # 建立 Conda 環境、啟用環境並安裝依賴套件
   conda env create -f environment.yml
   conda activate mlarena
   pip install -r requirements.txt

2. python train.py          → 產生 model.zip
3. 修改 run.py 的 STUDENT_ID
4. python run.py            → 上傳至競賽平台
```

## 觀測值與動作說明

| 項目 | 說明 |
|------|------|
| 觀測值 `observation` | `numpy.ndarray` shape `(8, 8, 111)` int8，棋盤狀態 |
| 動作遮罩 `action_mask` | `numpy.ndarray` shape `(4672,)` int8，值為 1 才合法 |
| 終局獎勵 | 勝 `+1`、負 `-1`、平 `0` |
| 評分 | 多局勝場數，越多越好 |
| 演算法 | 預設 MaskablePPO；可換成 MaskableA2C（修改 `model.py`） |

---

## 如何訓練更強的 Agent

### 1. Reward Shaping（獎勵塑形）

PettingZoo chess 預設只在終局給予 `+1 / -1 / 0`，中間過程沒有任何訊號，導致 Agent 難以學到具體策略。加入中間步獎勵可大幅加速學習：

| 策略 | 實作位置 | 說明 |
|------|----------|------|
| 子力差獎勵 | `train.py → _shape_reward()` 範例 1 | 以各棋種分值（P=1, N/B=3, R=5, Q=9）計算雙方差值，每步給予小額 reward |
| 存活步數獎勵 | `train.py → _shape_reward()` 範例 2 | 每步 +0.001，讓 Agent 學會不要輕易被將死 |

取消 `_shape_reward()` 中對應注釋即可啟用。

### 2. Self-Play（自對弈）

對手越強，你學越快。`train.py` 的 `step()` 中已有 Self-Play 骨架（注釋狀態）：

1. 先用隨機對手訓練至少 500k steps，產生 `model.zip`
2. 取消 `step()` 中 Self-Play 注釋，並在 `ChessSelfPlayEnv.__init__` 加入：
   ```python
   from model import ALGORITHM, SAVE_PATH
   self._opponent = ALGORITHM.load(SAVE_PATH)
   ```
3. 重新訓練——Agent 現在對抗自己的上一版本
4. 每隔 100k steps 覆蓋存檔，讓對手持續升級

### 3. 超參數調整

| 參數 | 預設值 | 效果 | 建議嘗試值 |
|------|--------|------|-----------|
| `learning_rate` | `3e-4` | 過大→震盪，過小→收斂慢 | `1e-4` ~ `1e-3` |
| `n_steps` | `2048` | 越大→梯度估計越穩定 | `1024` ~ `4096` |
| `batch_size` | `64` | 須整除 `n_steps × N_ENVS` | `128` ~ `256` |
| `ent_coef` | `0.01` | 越大→探索越多 | `0.02` ~ `0.05` |
| `TOTAL_TIMESTEPS` | `500_000` | 棋局複雜，建議 2M+ | `2_000_000`+ |

在 `train.py` 的 `ALGORITHM(...)` 呼叫中加入對應參數，詳細說明見 `model.py` 末尾注釋。

### 4. 網路結構

修改 `model.py` 的 `POLICY_KWARGS`，詳見該檔案末尾注釋。棋盤輸入維度為 7104，建議從 `[512, 512]` 開始探索。
