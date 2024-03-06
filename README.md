# MSU Super-Resolution Quality Metrics Benchmark

## Using for Super-Resolution Quality Metrics Dataset
* Put stats.json into "stats" directory
* Write "add_metric(<metric_name>, <device_name>)" to MetricsCheck.py for each metric you want calculate
* Write "calc_all_metrics(<force_to_recalcute_metric>, <repair_metrics' values>, <list \of datasets>)" to MetricsCheck.py

To install the dependencies into your conda environment with `python 3.9`, run:
```bash
pip install -r requirements.txt
```

To execute benchmark calculations run:
```bash
python MetricsCheck.py
```

To build correlation graphs run:
```bash
python BuildGraphs.py
```

## Using without Super-Resolution Quality Metrics Dataset

```python3
from utils import Metric, beautify_dict

#Show Metrics' List
print(list(beautify_dict.keys()))

metric_name = ""
device = "cuda:0"

gt_path = "gt_video.mp4"
sample_path = "sample_video.mp4"

# No-Reference Metric
estimator_nr = Metric("q-align_iqa", "cuda:0")
score, time = estimator_nr.predict(dist_video_path=sample_path)
print(f'Score of {estimator_nr.full_metric_name} is {score}')

# Full-Reference Metric
estimator_fr = Metric("topiq_fr", "cuda:1")
score, time = estimator_fr.predict(dist_video_path=sample_path, gt_video_path=gt_path)
print(f'Score of {estimator_fr.full_metric_name} is {score}')

# Metric with path to Frames
sample_frames_path = './frames/'
score, time = estimator_nr.predict(dist_frames_path=sample_frames_path)
print(f'Score of {estimator_fr.full_metric_name} is {score}')
```
