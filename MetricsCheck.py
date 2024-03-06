import os
import json

default_path = '.'
stats_path = os.path.join('.', 'stats', 'stats.json')

metrics_arr = []

def calc_all_metrics(force=False, repair_only=False, datasets=None):
    global metrics_arr
    last_correct_metric_values = [None for i in range(len(metrics_arr))]

    with open(stats_path, 'r') as f:
        full = json.load(f)
    for i in range(len(full)):
        if datasets is not None:
            if not full[i]['dataset'] in datasets:
                continue
        if full[i]['dataset'] in [ 'SR+Codecs', 'SR Dataset' ]:
            frames_path = full[i]['frames_path']
        else:
            frames_path = None
        if full[i]['dataset'] == 'SR+Codecs':
            for codec in full[i]['dist_videos']:
                for j in range(len(full[i]['dist_videos'][codec])):
                    calculated_this_video = False
                    for metric_index in range(len(metrics_arr)):
                        if (not repair_only) and (not force) and \
                            (metrics_arr[metric_index].full_metric_name in \
                            full[i]['dist_videos'][codec][j]['metrics'].keys()):
                            continue
                        if repair_only and ((last_correct_metric_values[metric_index] is None) \
                                or (full[i]['dist_videos'][codec][j]['metrics'][metrics_arr[metric_index].full_metric_name] !=  \
                                last_correct_metric_values[metric_index])):
                            last_correct_metric_values[metric_index] = full[i]['dist_videos'][codec][j]['metrics'][metrics_arr[metric_index].full_metric_name]
                            continue
                        
                        value, _ = metrics_arr[metric_index].predict(
                            preprocessed=False if not calculated_this_video else True,
                            dist_video_path=full[i]['dist_videos'][codec][j]['path'], 
                            gt_video_path=full[i]['path'], 
                            gt_frames_path=frames_path)
                        if not calculated_this_video:
                            calculated_this_video = True
                            for another_metric_index in range(len(metrics_arr)):
                                metrics_arr[another_metric_index].last_frames_cnt = metrics_arr[metric_index].last_frames_cnt
                        full[i]['dist_videos'][codec][j]['metrics'][metrics_arr[metric_index].full_metric_name] = value
                    with open(stats_path, 'w') as f:
                        json.dump(full, f, sort_keys = True, indent = 4)
        else:
            for j in range(len(full[i]['dist_videos'])):
                calculated_this_video = False
                for metric_index in range(len(metrics_arr)):
                    if (not repair_only) and (not force) and \
                        (metrics_arr[metric_index].full_metric_name \
                        in full[i]['dist_videos'][j]['metrics'].keys()):
                        continue
                    if repair_only and ((last_correct_metric_values[metric_index] is None) \
                            or (full[i]['dist_videos'][j]['metrics'][metrics_arr[metric_index].full_metric_name] !=  \
                            last_correct_metric_values[metric_index])):
                        last_correct_metric_values[metric_index] = full[i]['dist_videos'][j]['metrics'][metrics_arr[metric_index].full_metric_name]
                        continue
                    value, _ = metrics_arr[metric_index].predict(
                            preprocessed=False if not calculated_this_video else True,
                            dist_video_path=full[i]['dist_videos'][j]['path'],
                            gt_video_path=full[i]['path'],
                            gt_frames_path=frames_path)
                    if not calculated_this_video:
                        calculated_this_video = True
                        for another_metric_index in range(len(metrics_arr)):
                            metrics_arr[another_metric_index].last_frames_cnt = metrics_arr[metric_index].last_frames_cnt
                    full[i]['dist_videos'][j]['metrics'][metrics_arr[metric_index].full_metric_name] = value
                    with open(stats_path, 'w') as f:
                        json.dump(full, f, sort_keys = True, indent = 4)

def rename_metric(oldname, newname):
    with open(stats_path, 'r') as f:
        full = json.load(f)
    for i in range(len(full)):
        if full[i]['dataset'] == 'SR+Codecs':
            for codec in full[i]['dist_videos']:
                for j in range(len(full[i]['dist_videos'][codec])):
                    if not oldname in full[i]['dist_videos'][codec][j]['metrics'].keys():
                        continue
                    if newname in full[i]['dist_videos'][codec][j]['metrics'].keys():
                        full[i]['dist_videos'][codec][j]['metrics'].pop(oldname)
                    else:
                        full[i]['dist_videos'][codec][j]['metrics'][newname] = full[i]['dist_videos'][codec][j]['metrics'][oldname]
                        full[i]['dist_videos'][codec][j]['metrics'].pop(oldname)
        else:
            for j in range(len(full[i]['dist_videos'])):
                if not oldname in full[i]['dist_videos'][j]['metrics'].keys():
                    continue
                if newname in full[i]['dist_videos'][j]['metrics'].keys():
                    full[i]['dist_videos'][j]['metrics'].pop(oldname)
                else:
                    full[i]['dist_videos'][j]['metrics'][newname] = full[i]['dist_videos'][j]['metrics'][oldname]
                    full[i]['dist_videos'][j]['metrics'].pop(oldname)
    with open(stats_path, 'w') as f:
        json.dump(full, f, sort_keys = True, indent = 4)

from utils import Metric

def add_metric(metric_name, device):
    global metrics_arr
    metrics_arr.append(Metric(metric_name, device))

add_metric('liqe', 'cuda:3')
add_metric('liqe-mix', 'cuda:3')

print('All metrics added')

calc_all_metrics(force=True, datasets=['VSR_Benchmark'])
