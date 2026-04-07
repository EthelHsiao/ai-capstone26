# `load.py` 修改統計（Habitat `orientation` 型別修正）

## 修改目的
修正執行 `python load.py -f 1` 時的錯誤：

`TypeError: ... SensorSpec ... orientation ... expects _magnum.Vector3`

## 修改檔案
- `load.py`

## 修改摘要
1. 新增 `magnum` 匯入
2. 將三個 sensor 的 `orientation` 從 `list` 改為 `mn.Vector3(...)`
3. 將 `settings["sensor_pitch"]` 明確轉成 `float`，避免型別歧異

## 逐項差異

### 1) 匯入區
原本：
```python
import habitat_sim
```

修改後：
```python
import habitat_sim
import magnum as mn
```

### 2) `rgb_sensor_spec.orientation`
原本：
```python
rgb_sensor_spec.orientation = [
    settings["sensor_pitch"],
    0.0,
    0.0,
]
```

修改後：
```python
rgb_sensor_spec.orientation = mn.Vector3(
    float(settings["sensor_pitch"]),
    0.0,
    0.0,
)
```

### 3) `depth_sensor_spec.orientation`
原本：
```python
depth_sensor_spec.orientation = [
    settings["sensor_pitch"],
    0.0,
    0.0,
]
```

修改後：
```python
depth_sensor_spec.orientation = mn.Vector3(
    float(settings["sensor_pitch"]),
    0.0,
    0.0,
)
```

### 4) `semantic_sensor_spec.orientation`
原本：
```python
semantic_sensor_spec.orientation = [
    settings["sensor_pitch"],
    0.0,
    0.0,
]
```

修改後：
```python
semantic_sensor_spec.orientation = mn.Vector3(
    float(settings["sensor_pitch"]),
    0.0,
    0.0,
)
```

## 變更數量統計
- 修改檔案數：`1`
- 新增檔案數：`1`（本統計文件）
- `load.py` 主要修改點：`4`（`1` 個 import + `3` 個 orientation 指派）

## 相容性說明
這次修改不改變資料收集邏輯，只是把 `orientation` 參數改成目前 `habitat_sim` 綁定所要求的型別（`_magnum.Vector3`）。
