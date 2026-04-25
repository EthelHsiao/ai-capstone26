# HW3 實驗記錄 & 參數調整報告

> 本文檔記錄所有參數調整、測試案例和結果。
> 用於 Report 的 "Results & Discussion" 部分。
> **邊測試邊填** — 每次實驗後立即記錄結果。

---

## 📋 測試表單說明

每個實驗後，記錄以下信息：

- **起點座標** — 你點擊地圖時的 (x, y) 像素座標
- **目標物體** — rack / cooktop / sofa / cushion / stair
- **控制台輸出** — 複製 RRT 那行：`[RRT] Path found after X iterations, Y waypoints`
- **路徑平滑後** — 複製平滑那行：`[Smooth] Reduced path from X to Y waypoints`
- **導航結果** — ✓ 成功 / ✗ 失敗 / ⚠️ 警告
- **觀察備註** — 發生了什麼？速度如何？撞牆嗎？目標高亮了嗎？

---

## 🔬 實驗 1: 預設參數基準測試

**目的:** 建立基準線，了解預設參數的表現

**參數設置:**
```python
# map_processor.py
MAP_RESOLUTION = 10
OBSTACLE_POINT_THRESHOLD = 4
OBSTACLE_INFLATE_RADIUS = 0
SEMANTIC_BLOCK_INFLATE_RADIUS = 2
DOOR_CARVES = [((62, 64), 1)]
HEIGHT_FILTER_LOW = 0.05
HEIGHT_FILTER_HIGH = 2.2

# main.py (plan_path 參數)
max_iter = 8000
step_size = 15
goal_bias = 0.15
goal_tolerance = 15

# navigator.py
MOVE_AMOUNT = 0.05
TURN_AMOUNT = 1.0
```

**測試日期:** ___________

### 測試案例

#### 案例 1.1: Rack (近距離)

```
起點座標: (_____, _____)
目標: rack

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.

World path has _____ waypoints.
Spawning Agent at world position: (_____.3f, _____.3f)

Experiment summary:
target                 : rack
RRT result             : iterations=_____, nodes=_____, raw_waypoints=_____, smoothed_waypoints=_____
smoothing reduction    : _____%
smoothed path length   : _____ px / _____ m
straight-line distance : _____ m
occupancy              : free_regions=_____, obstacle_ratio=_____

Navigation summary:
frames=_____, target_seen_frames=_____, max_mask_pixels=_____
```

**導航結果:** ☐ 成功  ☐ 失敗  ☐ 警告

**觀察:**
- RRT 迭代數：_____________
- 路徑平滑效果：_____________
- 機器人運動：順暢 / 有點卡 / 撞牆
- 目標高亮：有 / 沒有
- 其他問題：_____________

---

#### 案例 1.2: Cooktop (中距離)

```
起點座標: (_____, _____)
目標: cooktop

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** ☐ 成功  ☐ 失敗  ☐ 警告

**觀察:**
- RRT 迭代數：_____________
- 路徑平滑效果：_____________
- 機器人運動：_____________
- 其他問題：_____________

---

#### 案例 1.3: Sofa (遠距離)

```
起點座標: (_____, _____)
目標: sofa

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** ☐ 成功  ☐ 失敗  ☐ 超時

**觀察:**
- 如果失敗：為什麼？（超時？無法找到路徑？）
- RRT 迭代數：_____________
- 其他問題：_____________

---

#### 案例 1.4: Cushion

```
起點座標: (_____, _____)
目標: cushion

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** ☐ 成功  ☐ 失敗  ☐ 警告

**觀察:**
- _____________

---

#### 案例 1.5: Stair

```
起點座標: (_____, _____)
目標: stair

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** ☐ 成功  ☐ 失敗  ☐ 警告

**觀察:**
- _____________

---

### 實驗 1 總結

**成功率:** ___/5 (___%)

**平均 RRT 迭代數:** _______

**主要問題:** 
```
☐ 沒有問題 — 預設參數表現很好
☐ RRT 太慢 — 某些情況超過 2000 iterations
☐ 機器人撞牆 — 需要增加 OBSTACLE_INFLATE_RADIUS
☐ 其他：_____________
```

**結論:**

_________________________________________________________________

_________________________________________________________________

---

## 🔬 實驗 2: 調整 Goal Bias

**目的:** 測試 goal_bias 對 RRT 收斂速度的影響

**改動:**
```python
# 只改這個，其他不變
goal_bias = 0.15 → 0.25
```

**測試日期:** ___________

### 對比測試

在實驗 1 失敗或很慢的案例上重新測試：

#### 案例 2.1: 原本慢的案例 (例: Sofa 遠距離)

```
起點座標: (_____, _____) [同實驗1]
目標: sofa

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** ☐ 成功  ☐ 失敗

**改善對比:**

| 指標 | 實驗1 (bias=0.15) | 實驗2 (bias=0.25) | 改善 |
|------|------------------|------------------|------|
| 迭代數 | _______ | _______ | +/- ______% |
| 路徑點數 | _______ | _______ | +/- ______% |
| 成功? | ✓/✗ | ✓/✗ | ☑ 改善 ☐ 變差 |

---

### 實驗 2 結論

**goal_bias 的影響:**

- bias = 0.15: _____________
- bias = 0.25: _____________

**建議:**
```
☐ 保持 0.15 (探索優先)
☐ 改成 0.20 (平衡)
☐ 改成 0.25 (收斂優先)
☐ 其他：_____________
```

---

## 🔬 實驗 3: 調整 Obstacle Inflation

**目的:** 測試障礙物膨脹半徑對導航安全性的影響

**改動:**
```python
OBSTACLE_INFLATE_RADIUS = 3 → 5
```

**測試日期:** ___________

### 觀察

#### 遭遇到的問題 (實驗1 中)

```
哪個案例機器人撞牆？
起點: (_____, _____)
目標: _____________
撞牆位置: _____________
```

#### 用新設置重新測試

```
起點座標: (_____, _____) [同上]
目標: _____________ [同上]

控制台輸出:
[RRT] Path found after _____ iterations, _____ waypoints.
[Smooth] Reduced path from _____ to _____ waypoints.
```

**導航結果:** 撞牆？ ☐ 有  ☐ 沒有

**路徑品質變化:**

| 指標 | 舊 (inflate=3) | 新 (inflate=5) | 變化 |
|------|----------------|----------------|------|
| 能通過? | _____ | _____ | ☑ 改善 ☐ 變差 |
| 迭代數 | _____ | _____ | +/- _______ |
| 通行性 | _____ | _____ | 更安全/更受限 |

---

### 實驗 3 結論

**OBSTACLE_INFLATE_RADIUS 的影響:**

- inflate = 3: _____________
- inflate = 5: _____________

**建議值:** _______

**理由:**

_________________________________________________________________

---

## 🔬 實驗 4: 其他參數調整 (可選)

如果你想試試 step_size 或 max_iter 的影響，可以在這裡記錄。

### 改動參數

```python
# 記錄你改了什麼
OLD: _____________
NEW: _____________
```

### 測試結果

```
案例: _____________
結果: ✓ 更好  ☐ 更差  ☐ 沒有顯著影響
迭代數: _____ → _____
備註: _____________
```

---

## 📊 最終推薦配置

根據所有實驗，選擇最佳參數組合：

```python
# map_processor.py
MAP_RESOLUTION = _____          # 原: 10
OBSTACLE_INFLATE_RADIUS = _____  # 原: 3

# main.py
max_iter = _____                # 原: 8000
step_size = _____               # 原: 15
goal_bias = _____               # 原: 0.15
goal_tolerance = _____          # 原: 15
```

---

## 📈 實驗數據彙總表

用這個表格總結所有實驗的關鍵數據（供 Report 使用）：

### 成功率對比

| 參數組合 | Rack | Cooktop | Sofa | Cushion | Stair | 總成功率 |
|---------|------|---------|------|---------|-------|----------|
| 預設 (bias=0.15, inflate=3) | ___ | ___ | ___ | ___ | ___ | ____% |
| 改進後 (bias=???, inflate=???) | ___ | ___ | ___ | ___ | ___ | ____% |

### 性能對比

| 指標 | 預設配置 | 改進配置 |
|------|---------|---------|
| 平均 RRT 迭代數 | _____ | _____ |
| 平均路徑點數 | _____ | _____ |
| 平均路徑平滑率 | ____% | ____% |
| 導航無碰撞? | 是/否 | 是/否 |

---

## 💬 Report Discussion 的重點

根據實驗結果，討論以下：

### Q1: Step Size 和 Goal Bias 的影響

**從實驗中發現:**

1. **Goal Bias 影響：**
   - 當 bias = 0.15 時：_____________
   - 當 bias = 0.25 時：_____________
   - **結論：** _____________

2. **Step Size 影響：**
   - 當 step_size = 15 時：_____________
   - 當 step_size = 20 時：_____________
   - **結論：** _____________

**與理論的對應:**
- RRT 預期行為是什麼？
- 我們的實驗是否符合預期？
- 為什麼會出現 (或沒出現) 某些現象？

---

### Q2: 實世界應用的挑戰

基於我們在模擬中遇到的問題，列舉實世界的困難：

1. **模擬中的問題 → 實世界挑戰**
   - 模擬: 完美的地圖，無雜訊
   - 實世界: 地圖誤差、動態障礙、感測器雜訊
   
2. **我們預設了什麼？**
   - 靜態環境
   - 完美的座標轉換
   - 無感測器誤差
   
3. **實世界需要額外處理**
   - _____________
   - _____________
   - _____________

---

## 🛠 Debug / Fix Log

> 每次發現 bug 或調整策略後，把「現象 → 原因 → 解法 → 驗證方式」記在這裡。
> 這段可以直接整理成 report 的 Results & Discussion / Implementation Notes。

### 1. Stair 目標找不到

**現象**

輸入 `stair` 時出現：

```text
ValueError: No valid pixels found for 'stair'.
```

**原因**

原本 `stair` semantic color 填成 `[255, 31, 0]`，這其實不是 stair；點雲中該顏色沒有 stair pixels。

**解決方法**

將 stair 顏色改成 101-category color map 中的正確顏色：

```python
"stair": [[173, 255, 0]]
```

並將 Habitat semantic object id 改成：

```python
"stair": 192
```

**驗證**

修正後 semantic map 可找到：

```text
stair 158 pixels
```

**需要保留的 log / 截圖**

- `Target locations` 表格中 `stair pixels > 0`
- 一張 stair 的 RRT path screenshot

---

### 2. Cushion 顏色與 mask id 錯誤

**現象**

`cushion` 與 `sofa` 很接近，且一開始 cushion 的 semantic/color 和 Habitat mask 不穩定。

**原因**

原本 cushion color `[255, 5, 153]` 對應到其他類別，不是 cushion。  
另外 cushion 在 Habitat 中有多個 object instances，不能只用單一 id。

**解決方法**

將 cushion color 改為：

```python
"cushion": [[255, 9, 92]]
```

將 mask id 改為多個 cushion instances：

```python
"cushion": [431, 430, 400, 350, 149, 151, 205, 251, 223]
```

並在 `navigator.py` 用 `np.isin()` 支援多個 target ids。

**驗證**

```text
cushion 21 pixels
Target mask visible in X/Y frames (max Z pixels)
```

**需要保留的 log / 截圖**

- `cushion pixels`
- Habitat RGB 視窗中紅色 mask 截圖

---

### 3. Semantic map 顯示太小、黑底且與老師範例差很多

**現象**

OpenCV 顯示的 map 很小，而且黑底讓空白區看起來不像老師範例。

**原因**

實際 map 解析度只有約 `104 x 159` pixels，且原始 display 直接用黑色表示 empty/free space。

**解決方法**

新增：

```python
DISPLAY_SCALE = 3
semantic_map_to_uint8(..., white_background=True)
```

顯示時放大，但滑鼠點擊會轉回原始 map coordinate。  
另外移除完全孤立的 display-only semantic pixels，減少空白處雜點。

**驗證**

```text
map (159, 104, 3)
display (477, 312, 3)
black_pixels_after_display_conversion 0
```

**需要保留的 log / 截圖**

- 修正後的 semantic map screenshot
- RRT path 視窗 screenshot

---

### 4. RRT path 線條太粗、Start/Goal 文字擋住地圖

**現象**

Path 視覺化時綠線與 `Start` / `Goal` 文字很粗，放大後看起來像像素塊。

**原因**

原本先在低解析度 map 上畫線與文字，再用 nearest-neighbor 放大，導致鋸齒與文字過大。

**解決方法**

改成：

1. 先放大 semantic map。
2. 再用 display coordinate 畫 path / waypoint / Start / Goal。
3. 使用 `cv2.LINE_AA` 抗鋸齒。

**驗證**

RRT Path 視窗線條變細、文字不再遮住地圖。

**需要保留的 log / 截圖**

- 修正前後 path visualization screenshot 對比

---

### 5. Start 或 goal 可能在 obstacle 上

**現象**

若使用者點到牆或物件上，RRT 可能卡住或無法向外長 tree。

**原因**

RRT collision check 會檢查起點所在 pixel；若起點是 obstacle，所有從 root 延伸出去的線段都會失敗。

**解決方法**

選 start 後先呼叫 `_nearest_free_pixel()`，若 start 在 obstacle 上，改用附近最近 free pixel。

**驗證**

Console 會印：

```text
Start pixel was on an obstacle; using nearest free pixel ...
```

**需要保留的 log / 截圖**

- 若發生，保留該行 log

---

### 6. Goal 不能直接設在物體本體上

**現象**

若 goal 直接使用 semantic target pixel，會落在物體/牆上，RRT 不能走到該點。

**原因**

Target object pixels 通常是 obstacle；spec 要求 target point 應在物體前方或附近可走區。

**解決方法**

`pick_goal()` 流程：

1. `get_goal_pixels()` 找出 target object 本體 pixels。
2. 建立 free-space connected components。
3. 從 target 周圍找 free pixel。
4. 優先選與 start 同一 connected component 且符合 target-side / visibility 條件的點。

**目前說法**

```text
The RRT goal is selected as a reachable free-space point near the semantic target,
instead of the target object pixel itself.
```

**需要保留的 log / 截圖**

- `Start pixel: ..., Goal pixel: ...`
- `Target locations` 表格，確認 goal 靠近 target centroid

---

### 7. Rack goal 會被錯放到隔壁房間牆邊

**狀態:** 已避免假成功，但需搭配後續 connectivity / floor-free-space 修正。

**現象**

從右下角房間找 `rack` 時，因 rack 貼牆，程式曾把 goal 放到右下角房間牆邊。路徑沒有穿牆，但其實走到錯的房間側，看不到 rack。

**原因**

舊版 goal selection 只找最近 reachable free pixel。Rack 在牆另一側貼牆，隔牆的 free pixel 在 2D 距離上很近，因此被誤選。

**解決方法**

新增 `TARGET_GOAL_DIRECTIONS`，對 rack 偏好/限制 goal 在正確房間側：

```python
TARGET_GOAL_DIRECTIONS = {
    "rack": (0.0, -1.0),
}
```

若找不到 rack 正確側可走點，不再 fallback 到錯房間，而是印出 error，避免假成功。

**嘗試歷程**

1. 嘗試用 nearest reachable free pixel：失敗，會選到隔牆的右下角房間側。
2. 嘗試 line-of-sight / blocker map：改善有限，因 rack 貼牆且 semantic projection 太近。
3. 改成 rack 使用 preferred side：可避免 goal 被放到錯側。
4. 但 preferred side 一開始導致 disconnected error，轉而處理 occupancy connectivity。

**驗證**

修正後 rack goal 會選到：

```text
goal (68, 84)
```

**需要保留的 log / 截圖**

- 右下角 start 到 rack 的 `Start pixel / Goal pixel`
- RRT path screenshot，確認 goal 在 rack 正確側

---

### 8. Rack 正確側與右下角房間被假障礙物切斷

**狀態:** 以局部 door carve 解決該門口連通，但仍需配合後續 floor-free-space 檢查整體地圖。

**現象**

Rack goal 改到正確側後，從右下角 start 會出現：

```text
[Goal] ERROR: no reachable goal found on the preferred side of 'rack'.
The target-side room is disconnected in the current occupancy map.
```

**原因**

點雲投影在真實門口附近留下少量 obstacle pixels，導致右下角房間和 rack 正確側被切成不同 connected components。

**解決方法**

只在確認過的門口位置做局部 carve，而不是放寬整張 occupancy map：

```python
DOOR_CARVES = [((62, 64), 1)]
```

**嘗試歷程**

1. 嘗試將 `OBSTACLE_POINT_THRESHOLD` 調到 5：rack component 可連通，但副作用是 stair / 家具障礙變得太鬆。
2. 改成只 carve 確認過的門口 pixel `(62, 64)`：較局部、較可解釋。
3. 後續又改成 floor-based free-space，仍保留此 door carve 作為點雲假障礙修正。

**驗證**

```text
start/rack labels 1 1 same True
target rack goal (68, 84)
```

**需要保留的 log / 截圖**

- `same True` 的 connectivity check log
- RRT path screenshot，確認從右下角可到 rack 正確側

---

### 9. RRT 在窄通道中可能找不到路

**狀態:** 加入 fallback 後可避免 demo 卡死；但 report 需說明主方法仍是 RRT，fallback 是 robustness improvement。

**現象**

即使 occupancy map 已連通，RRT 仍可能輸出：

```text
[RRT] Failed to find a path within the iteration budget.
Planner could not find a path.
```

**原因**

RRT 是隨機演算法，在窄通道或長距離環境中可能需要更多 iterations 才能成功。Connectivity 存在不代表 RRT 一定能在固定 budget 內找到。

**解決方法**

加入 fallback：

```text
RRT first, grid fallback only if RRT fails.
```

Fallback 使用 8-connected grid search 找出保底路徑，之後仍然套用 `smooth_path()` 和相同的 Habitat navigation。

**驗證**

```text
[RRT] Falling back to grid search for this connected but narrow route.
[Grid] Fallback path found with 138 waypoints.
[Smooth] Reduced path from 138 to 14 waypoints.
```

**Report 寫法**

可寫成 robustness / bonus improvement：

```text
When RRT fails in a narrow connected region, a grid-based fallback is used
to recover a valid waypoint path before smoothing.
```

**需要保留的 log / 截圖**

- RRT fail + grid fallback + smoothing log
- fallback 後的 RRT Path screenshot

---

### 10. Path 會穿過 stair / 樓梯

**狀態:** 已加入 stair semantic blocker；目前 stair pixels 都會被強制視為 obstacle。

**現象**

路徑曾穿過樓梯區域，但作業只使用 first floor，樓梯往樓上不應當作可走區。
![alt text](image-1.png)
**原因**

為了打通門口，`OBSTACLE_POINT_THRESHOLD` 曾調高到 5，使稀疏的 stair projection pixels 沒有被視為 obstacle。

**解決方法**

新增 stair semantic blocker：不管 threshold 多寬鬆，stair 都強制為 obstacle，並膨脹 2 pixels。

```python
STAIR_COLOR = np.array([173, 255, 0])
SEMANTIC_BLOCK_INFLATE_RADIUS = 2
```

**驗證**

```text
stair pixels 158 occupied_pixels 158
```

**嘗試歷程**

1. 為了連通 rack，曾把 `OBSTACLE_POINT_THRESHOLD` 調到 5。
2. 發現 path 會穿過 stair，代表 threshold 過度放寬了樓梯區。
3. 加入 stair semantic blocker，使 stair 不受 threshold 放寬影響。
4. 後續若 path 仍靠近 stair，需要調整 `SEMANTIC_BLOCK_INFLATE_RADIUS`。

**需要保留的 log / 截圖**

- stair occupied check log
- path 不穿過 stair 的 screenshot

---

### 11. Path 看起來壓到家具 / 障礙物

**狀態:** 尚未完全解決；目前先用 occupancy overlay 判斷問題來源，下一步要分辨是 occupancy 太鬆還是 smoothing 太激進。

**現象**

RRT Path 視窗中，路線看起來會壓到 semantic map 上的彩色物件。
![alt text](image.png)

**原因**

Semantic map 是給人看的彩色投影；RRT 實際使用的是 occupancy map。當 `OBSTACLE_POINT_THRESHOLD` 太高時，部分稀疏家具投影會被 occupancy 視為 free，因此 path 會壓到彩色物件。

**解決方法**

將：

```python
OBSTACLE_POINT_THRESHOLD = 5
```

調整為更保守但仍能連到 rack 的：

```python
OBSTACLE_POINT_THRESHOLD = 4
```

並在 RRT Path 視窗加上 occupancy overlay，將 planner 視為 obstacle 的 pixels 以淡灰色顯示。

**嘗試歷程**

1. `OBSTACLE_POINT_THRESHOLD = 5`：rack 較容易連通，但 path 可能走上 stair / 家具。
2. 加入 stair blocker：解決樓梯不可走問題。
3. 改成 `OBSTACLE_POINT_THRESHOLD = 4`：比 5 更保守，仍可保持 rack connectivity。
4. 發現圖上仍可能看似壓到彩色 semantic object，因此新增 occupancy overlay。
5. 尚待確認：若綠線穿過灰色 overlay，代表 collision / smoothing 有問題；若只穿過彩色但非灰色，代表 occupancy map 認為該處可走。

**驗證**

```text
threshold 4 start/rack same True regions 48 obs 0.201
goal (68, 84)
```

**如何判讀截圖**

- 綠線壓到灰色 overlay：collision / path smoothing 可能有問題。
- 綠線壓到彩色但非灰色：occupancy map 覺得該處可走，需調 threshold 或加入 semantic blocker。

**需要保留的 log / 截圖**

- 帶 occupancy overlay 的 RRT Path screenshot
- 若仍壓障礙，貼上該 screenshot 和 console summary
- 下一次請貼：同一 start/target 下「raw path 不 smoothing」和「smoothing 後」的對比 screenshot

---

### 12. 門口不明顯 / planner 不知道哪裡可以走

**狀態:** 目前已改成 floor-based free-space，但仍需更多 start/target 實測確認是否會造成 free-space 過胖或 smoothing shortcut。

**現象**

Semantic map 看起來不像老師範例，門口不明顯；path 也可能壓到彩色物件或被假牆堵住。

**原因**

舊版 occupancy 定義是：

```text
沒有牆/家具投影點的空白 = free
```

這會造成兩個問題：

1. Unknown empty space 也可能被當成 free。
2. 門口是否打開取決於牆/家具投影點是否剛好堵住，而不是看地板是否連續。

老師範例中門很清楚，是因為保留 floor points 時，房間和門口的地板連續性很明顯。

**解決方法**

改成 floor-based free-space：

```text
先用 floor-colored points 建 floor_free
只有 floor_free 覆蓋到的地方才可走
再把牆、家具、stair 等 obstacle 從 floor_free 中扣掉
```

相關參數：

```python
USE_FLOOR_FREE_SPACE = True
FLOOR_FREE_DILATE_RADIUS = 4
OBSTACLE_POINT_THRESHOLD = 4
DOOR_CARVES = [((62, 64), 1)]
```

RRT 視覺化也新增 free-space tint，讓 planner 認為能走的地板區域更清楚。

**嘗試歷程**

1. 舊方法：`沒有 obstacle projection = free`。問題是 unknown empty space 也可能被當成 free，門口受投影雜訊影響。
2. 改成 floor-based free-space：`有 floor point 的地方才是候選 free`，再扣掉 obstacle。
3. 初始 `FLOOR_FREE_DILATE_RADIUS = 2` 太保守，free regions 過多，右下角到 rack 仍不連通。
4. 掃描半徑後發現 `FLOOR_FREE_DILATE_RADIUS = 4` 是第一個能讓右下角到 rack 正確側連通的值。
5. 目前保留 `DOOR_CARVES = [((62,64),1)]` 修正局部假障礙。
6. 尚待確認：free-space tint 是否過胖、path 是否因 smoothing 穿過物件邊緣。

**驗證**

目前測試：

```text
floor_free True dilate 4 regions 56 obs 0.538 start/rack same True
rack goal (68, 84) occ 0 region 2
```

五個目標從右下角測試都能找到 goal：

```text
rack goal (68, 84)
cooktop goal (28, 46)
sofa goal (57, 138)
cushion goal (31, 142)
stair goal (72, 108)
```

**需要保留的 log / 截圖**

- 帶 free-space tint 和 occupancy overlay 的 RRT Path screenshot
- `floor_free True dilate 4 ... start/rack same True`
- 若 path 仍穿過灰色 obstacle，貼上 screenshot
- 下一次請貼：同一張 path 圖，並說明綠線是否穿過灰色 overlay 還是只穿過彩色 semantic object

---

### 13. Habitat navigation mask 有時看不到目標

**狀態:** 已加入 mask 統計；若仍常為 0，後續需改成導航結束後轉向 target centroid，而不是原地掃描。

**現象**

Agent path execution completed，但 target mask 可能是：

```text
target_seen_frames=0
```

**原因**

紅色 mask 是在 Habitat RGB camera view 上由 semantic sensor 產生。只有當相機當下看得到該 object id 時才會出現。即使 RRT 到達目標附近，也可能因為最後朝向不對、物件太小或被遮擋而看不到 mask。

**解決方法**

新增 navigation stats：

```text
Navigation summary: frames=..., target_seen_frames=..., max_mask_pixels=...
```

並讓 cushion 支援多個 semantic ids。  
目前也曾加入原地掃描，但 demo 看起來可能不自然，後續可改成「最後轉向 target centroid」。

**需要保留的 log / 截圖**

- `target_seen_frames > 0` 的 log
- RGB 視窗中紅色透明 mask 截圖

---

### 14. 8 個門口被假牆完全堵死，導致多個房間孤立無法到達

**狀態:** 已解決 — DOOR_CARVES 擴展到 9 個門口，所有 5 個目標從任意起點均可到達。

**現象**

從右下角房間出發選 `rack` 時出現：

```text
[Goal] ERROR: no reachable goal found on the preferred side of 'rack'.
The target-side room is disconnected in the current occupancy map.
```

用 connected component 分析發現：地圖有 44–56 個孤立 free region，原本的 DOOR_CARVES 只挖了 1 個門口 `((62, 64), 1)`，其他 8 個門口全部堵死。對照助教範例（整棟公寓一個連通區），差距很明顯。

**原因**

點雲投影在每個真實門口位置，各高度層（0–2.2m）均有大量投影點（每 0.2m 層 100–500 個），原因是：

1. 門框側柱整個高度都有掃描點
2. 鄰室牆面從掃描角度透過門口投影到門口 pixel
3. 點雲密度在 threshold=4 下，門口 1–2 個 pixel 就夠被標為 obstacle

這是 occupancy map 問題，不是 RRT 或 goal selection 的問題。

**嘗試歷程**

1. **嘗試調低 `HEIGHT_FILTER_HIGH`**（2.2 → 2.0 → 1.8 → 1.6m），假設門框頂部是主要阻擋來源。
   - 結果：2.2m→2.0m 省 1 個 carve，2.0m→1.8m 再省 1 個，1.8m 以下不再改善。
   - 失敗原因：門口點雲不只在頂部，整個高度都密集，height filter 無法根治。

2. **改用 connected component 自動偵測所有孤立房間**，找每個房間邊界到主區域邊界的最近點對，得到所有需要 carve 的門口中間點。
   - 結果：找到 8 個孤立 region 距主區域只有 2px（= 1 個障礙 pixel 的假牆）。
   - 此方向成功，確認位置後寫入 `DOOR_CARVES`。

**解決方法 / 目前做法**

在 `map_processor.py` 的 `DOOR_CARVES` 新增全部 9 個門口：

```python
DOOR_CARVES = [
    ((50, 15), 2),   # top room ↔ main
    ((81, 21), 2),   # upper-right pocket ↔ main
    ((68, 44), 2),   # upper-right room ↔ main
    ((19, 58), 2),   # left corridor ↔ main
    ((62, 64), 2),   # central doorway (原有，radius 1→2)
    ((55, 93), 2),   # rack room ↔ main
    ((73,112), 2),   # rack room lower ↔ main
    ((94,142), 2),   # bottom-right room ↔ main
    ((56,147), 2),   # bottom strip ↔ main
]
```

執行時機：occupancy map 全部建完之後，最後用 `cv2.circle(occupancy_map, (cx,cy), r, 0, -1)` 強制把門口像素設為 free，不受 threshold / inflation 影響。

**驗證**

```text
Connected free regions: 16  (主區域 area=5690px)
rack     dist=1px  ✓
cooktop  dist=4px  ✓
sofa     dist=1px  ✓
cushion  dist=2px  ✓
stair    dist=3px  ✓
```

從右下角房間出發到 rack 成功規劃路徑並導航。

**需要保留的 log / 截圖**

- Occupancy Map 視窗截圖（整棟公寓為單一紫色連通區域）
- 從右下角房間出發到 rack 的 `Start pixel / Goal pixel` log
- RRT Path 截圖確認路線穿越多個房間

---

### 15. OBSTACLE_INFLATE_RADIUS 0 → 1：減少家具邊緣 phantom 像素

**狀態:** 已解決 — inflate=1 搭配 DOOR_CARVES 同時保持連通且減少 phantom。

**現象**

Occupancy Map 視窗顯示大量藍紅色（phantom = 有語意顏色但 RRT 認為可走），估計 1544 個 phantom pixels（佔 free 空間 20%）。路徑規劃時看起來會壓到彩色家具邊緣。

**原因**

semantic map 用全部點著色，occupancy map 只用 height-filtered 且 count≥threshold 的點。家具邊緣投影點稀疏（1–3 個），低於 threshold=4，被視為 free，導致語意地圖上有顏色但 RRT 可穿過。

**嘗試歷程**

1. **inflate=0（原始）**：phantom=1544，但所有 5 目標連通。
2. **inflate=1，不加 DOOR_CARVES**：phantom 降到 591，但多個目標又斷線。
3. **inflate=1 + DOOR_CARVES**：phantom=591，5 目標全部連通。✓

**解決方法 / 目前做法**

```python
# map_processor.py
OBSTACLE_INFLATE_RADIUS = 1  # 原為 0
```

DOOR_CARVES 在 inflate 之後執行，把因 inflate 被縮窄的門口再挖回來。

**驗證**

```text
phantom pixels: 1544 → 591（減少 62%）
5 目標距離主區域: rack=1px, cooktop=4px, sofa=1px, cushion=2px, stair=4px
```

**需要保留的 log / 截圖**

- 加 inflate=1 前後的 Occupancy Map 視窗對比截圖（藍紅色減少）
- 同一 start/target path，確認路線不再明顯壓到家具邊緣

---

### 16. Rack/Stair 右側走道與主區域仍隔著 1.4 px 假牆

**狀態:** 已解決 — 新增第 10 個 DOOR_CARVE `((85, 59), 1)`，15 個 carve 後 5 目標均可從任意起點到達。

**現象**

加入 9 個 DOOR_CARVES（Issue #14）並驗證可達性後，對照 HW3 spec 提供的公寓俯瞰圖，發現 rack 所在的右側走道（R5, x[87..97], z[56..103]）與上方相鄰房間之間仍然是孤立 region：

```text
R5 pix (87..97), zpix (56..103) area=134  ← 孤立
nearest gap to main: dist=1.4px  at (85,59)/(86,59)
Connected free regions: 16  main area=5690px
```

**原因**

Issue #14 的 9 個 DOOR_CARVES 打通了 rack room 和主區域的下方連接（`((55,93),2)`, `((73,112),2)`），但 R5（rack 右側走道，z=56..103）與上方房間的門口在 (85,59) 附近，未被涵蓋。門口點 (85,59) 只有 1.4px gap，被 inflate=1 的障礙膨脹填滿，需要單獨 carve。

**嘗試歷程**

1. 確認 9 個 carve 後仍有孤立 region：執行 connectedComponentsWithStats，發現 R5（area=134px）在主區域之外。
2. 掃描 R5 邊界和 main 邊界之間的最近點對：找到 (85,59)/(86,59)，gap = 1.4px。
3. 在 carve 之前先確認 (85,59) 原始值為 255（obstacle）：合法假牆，非誤刪。
4. 新增 `((85, 59), 1)` 到 DOOR_CARVES：半徑 1 剛好足以連通 1.4px gap。

**解決方法 / 目前做法**

在 `map_processor.py` 的 `DOOR_CARVES` 新增第 10 個門口：

```python
DOOR_CARVES = [
    ((50, 15), 2),   # top room ↔ main
    ((81, 21), 2),   # upper-right pocket ↔ main
    ((68, 44), 2),   # upper-right room ↔ main
    ((19, 58), 2),   # left corridor ↔ main
    ((62, 64), 2),   # central doorway
    ((85, 59), 1),   # rack/stair right strip ↔ main (1.4px gap)  ← 新增
    ((55, 93), 2),   # rack room ↔ main
    ((73, 112), 2),  # rack room lower ↔ main
    ((94, 142), 2),  # bottom-right room ↔ main
    ((56, 147), 2),  # bottom strip ↔ main
]
```

**驗證**

```text
Connected free regions: 15  (主區域 area=5787px)
rack     dist=1px  ✓
cooktop  dist=4px  ✓
sofa     dist=1px  ✓
cushion  dist=2px  ✓
stair    dist=3px  ✓
```

無右側孤立 region；所有 5 目標從任意起點均可到達。

**需要保留的 log / 截圖**

- 更新後的 Occupancy Map 截圖（整棟公寓為單一主要連通區域）
- 從右側房間 start 到 rack 的成功 RRT 路徑截圖
- `Connected free regions: 15` 的 console log

---

## 🎯 額外觀察 (可選)

有沒有發現什麼有趣的現象或 bug？記在這裡：

- **現象 1:** _____________
- **現象 2:** _____________
- **可能的原因:** _____________

---

## ✅ 實驗完成檢查清單

- [ ] 實驗 1 (預設配置) 完成
- [ ] 實驗 2 (Goal Bias) 完成
- [ ] 實驗 3 (Obstacle Inflation) 完成
- [ ] 最終配置選定
- [ ] 數據彙總表填完
- [ ] Report Discussion 要點整理好
- [ ] 取截圖（路徑、導航窗口）

---

## 📸 截圖清單

列出你應該取的截圖（供 Report 用）：

- [ ] 預設配置的路徑可視化窗口
- [ ] 改進配置的路徑可視化窗口
- [ ] Habitat RGB 導航窗口（顯示目標高亮）
- [ ] 對比圖（預設 vs 改進）

---

## 🔗 相關文件

- `CLAUDE.md` — 技術背景 & spec
- `TUNING.md` — 參數詳細參考
- `README.md` — 執行和排除故障指南
- `main.py` — RRT 實作代碼

---

**最後更新:** ___________  
**實驗者:** ___________  
**狀態:** ☐ 進行中  ☐ 完成
