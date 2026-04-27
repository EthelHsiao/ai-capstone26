# HW3 Robot Navigation

This project builds a 2D semantic/occupancy map from the `apartment_0` semantic
point cloud, plans a route with RRT, and executes the waypoint path in
Habitat-Sim while highlighting the target object.

## Files

- `main.py` - main entry point, RRT planner, goal selection, visualization
- `map_processor.py` - point cloud processing and occupancy map construction
- `navigator.py` - Habitat-Sim setup and waypoint execution
- `semantic_3d_pointcloud/` - required point cloud data
- `report.md` - implementation details and experiment discussion

## Requirements

Use the provided Habitat environment:

```bash
conda activate habitat
```

The Habitat Replica scene is expected at:

```text
../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply
```

The point cloud files should exist under:

```text
semantic_3d_pointcloud/point.npy
semantic_3d_pointcloud/color0255.npy
```

## Run

From this directory:

```bash
conda activate habitat
python main.py
```

The program will:

1. build the semantic map and occupancy map,
2. show the map window,
3. ask you to click a start point,
4. ask for a target name,
5. run RRT and path smoothing,
6. visualize the planned path,
7. execute the path in Habitat.

Valid target names:

```text
rack
cooktop
sofa
cushion
stair
```

In the start-selection window, click a start position on the map. Press `q` to
quit without selecting.

## Optional Experiments

Some experiment scripts may be present depending on the branch:

```bash
python experiment.py
python experiment_targets.py
python experiment_smoothing.py
```

They generate path images and Markdown tables under `experiments/`.

## Notes

- Pixel coordinates are map coordinates in `(x, y)` order.
- Habitat/world waypoints are converted to `(x, z)` coordinates.
- The final map uses floor-based free space, 10 cm obstacle inflation, and a
  1.5 m height filter to keep doorways open while avoiding furniture.
