import matplotlib.pyplot as plt
from highcharts import Highchart as hc
from numpy import random
import os
import json
import scipy
import numpy as np

mxcnt = 31

default_path = '.'
stats_path = os.path.join(default_path, "stats", "stats.json")
img_path = os.path.join(default_path, "graphs")
corstats_path = os.path.join(default_path, 'stats', 'corstats.json')
CIstats_path = os.path.join(default_path, 'stats', 'CIstats.json')

metrics = []
reverse_metrics = [
    "NIQE (NR; over Y)",
    "NIQE OpenCL0 (NR; over Y)",
    "MSAD (FR; over Y)",
    "MSE (FR; over Y)",
    "VQM (FR; over Y)",
    "LPIPS (Alex) (FR; over RGB)",
    "LPIPS (VGG) (FR; over RGB)",
    "DISTS (FR; over RGB)",
    "A-DISTS (FR; over RGB)",
    "PieAPP (FR; over RGB)",
    'PI (NR; over RGB)',
    'BRISQUE (NR; over RGB)',
    "ST-LPIPS (Alex) (FR; over RGB)",
    "ST-LPIPS (VGG) (FR; over RGB)",
    "ILNIQE (NR; over RGB)"
]

firstTime = True

def weigh_function(values, length = 0):
    if length == 0:
        length = len(values)
    for i in range(len(values)):
        if np.isnan(values[i]):
            values[i] = 0.
    l = np.mean(values) - (1.96 * np.std(values)) / ((length + 0)**0.5)
    u = np.mean(values) + (1.96 * np.std(values)) / ((length + 0)**0.5)
    if l > u:
        tmp = l
        l = u
        u = tmp
    if l < 0.:
        l = 0.
    if l > 1.:
        l = 1.
    if u < 0.:
        u = 0.
    if u > 1.:
        u = 1.
    return list([l, u])
    

def hcplot(table_name, corp_arr, cors_arr, cork_arr, pCI, sCI, kCI, mn, mx, path = None, more_info = None):
    def find(arr, obj):
        for i in range(arr.shape[0]):
            if arr[i] == obj:
                return i
        return -1
    def rnd(arr, prec):
        for i in range(len(arr)):
            arr[i] = round(arr[i], prec)
        return arr
    corp_arr = np.array(corp_arr)
    cors_arr = np.array(cors_arr)
    cork_arr = np.array(cork_arr)
    pCI = np.array(pCI)
    sCI = np.array(sCI)
    kCI = np.array(kCI)
    
    corp_arr_b = []
    cors_arr_b = []
    cork_arr_b = []
    pCI_b = []
    sCI_b = []
    kCI_b = []
    metrics_arr = []
    for i in range(len(metrics)):
        print(metrics[i], corp_arr[i], cors_arr[i], cork_arr[i])
        if min(corp_arr[i], cors_arr[i], cork_arr[i]) > -1:
            metrics_arr.append(metrics[i])
            corp_arr_b.append(corp_arr[i])
            cors_arr_b.append(cors_arr[i])
            cork_arr_b.append(cork_arr[i])
            pCI_b.append(pCI[i])
            sCI_b.append(sCI[i])
            kCI_b.append(kCI[i])
    corp_arr = np.array(corp_arr_b)
    cors_arr = np.array(cors_arr_b)
    cork_arr = np.array(cork_arr_b)
    pCI = np.array(pCI_b)
    sCI = np.array(sCI_b)
    kCI = np.array(kCI_b)
    
    d = []
    for i in range(len(metrics_arr)):
        d.append((cors_arr[i], metrics_arr[i]))
    d.sort()
    corp_sort = []
    cors_sort = []
    cork_sort = []
    pCI_sort = []
    sCI_sort = []
    kCI_sort = []
    
    metrics_sort = []
    metrics_arr = np.array(metrics_arr)
    for (mean, metr) in d:
        metrics_sort.append(metr)
        idx = find(metrics_arr, metr)
        corp_sort.append(corp_arr[idx].tolist())
        cors_sort.append(cors_arr[idx].tolist())
        cork_sort.append(cork_arr[idx].tolist())
        pCI_sort.append(list(pCI[idx]))
        sCI_sort.append(list(sCI[idx]))
        kCI_sort.append(list(kCI[idx]))
    
    chart = hc()
    metrics_sort.append("Subjective Score")
    metrics_sort.reverse()
    corp_sort.append(1)
    corp_sort.reverse()
    cors_sort.append(1)
    cors_sort.reverse()
    cork_sort.append(1)
    cork_sort.reverse()
    pCI_sort.append([1, 1])
    pCI_sort.reverse()
    sCI_sort.append([1, 1])
    sCI_sort.reverse()
    kCI_sort.append([1, 1])
    kCI_sort.reverse()
    options = {
        'chart' : { 'type' : 'bar', 'height' : '100%' },
        'title' : { 'text' : table_name },
        'legend' : { 'enabled' : True },
        'xAxis' : { 'categories' : metrics_sort },
        'yAxis' : { 'title' : { 'text' : 'Correlation with Confidence Intervals' } },
    }
    chart.set_dict_options(options)
    chart.add_data_set([round(val, 5) for val in corp_sort], 'bar', 'Pearson', id = '1', color = '#b95144')
    chart.add_data_set([round(val, 5) for val in cors_sort], 'bar', 'Spearman', id = '2', color = '#e7e8d2')
    chart.add_data_set([round(val, 5) for val in cork_sort], 'bar', 'Kendall', id = '3', color = '#a7beae')
    
    chart.add_data_set([[round(val1, 2), round(val2, 2)] for (val1, val2) in pCI_sort], 'errorbar', 'Pearsonn', linkedTo = '1')
    chart.add_data_set([[round(val1, 2), round(val2, 2)] for (val1, val2) in sCI_sort], 'errorbar', 'Spearman', linkedTo = '2')
    chart.add_data_set([[round(val1, 2), round(val2, 2)] for (val1, val2) in kCI_sort], 'errorbar', 'Kendall', linkedTo = '3')
    chart.save_file(path)
    with open(corstats_path, 'r') as f:
        resdict = json.load(f)
    with open(CIstats_path, 'r') as f:
        CIresdict = json.load(f)
    for i in range(len(metrics_sort)):
        if not metrics_sort[i] in resdict.keys():
            resdict[metrics_sort[i]] = {}
        if not metrics_sort[i] in CIresdict.keys():
            CIresdict[metrics_sort[i]] = {}
        if not table_name in resdict[metrics_sort[i]].keys():
            resdict[metrics_sort[i]][table_name] = {}
        if not table_name in CIresdict[metrics_sort[i]].keys():
            CIresdict[metrics_sort[i]][table_name] = {}
        resdict[metrics_sort[i]][table_name]['Pearson'] = corp_sort[i]
        resdict[metrics_sort[i]][table_name]['Spearman'] = cors_sort[i]
        resdict[metrics_sort[i]][table_name]['Kendall'] = cork_sort[i]
        CIresdict[metrics_sort[i]][table_name]['Pearson'] = [pCI_sort[i][0], pCI_sort[i][1]]
        CIresdict[metrics_sort[i]][table_name]['Spearman'] = [sCI_sort[i][0], sCI_sort[i][1]]
        CIresdict[metrics_sort[i]][table_name]['Kendall'] = [kCI_sort[i][0], kCI_sort[i][1]]
    with open(corstats_path, 'w') as f:
        json.dump(resdict, f, sort_keys = True, indent = 4)
    with open(CIstats_path, 'w') as f:
        json.dump(CIresdict, f, sort_keys = True, indent = 4)
    
    
def get_correlation(video, path = None, cor_type = None):
    def get_info(video):
        SI = video['SI']
        TI = video['TI']
        rX = video['resolutionX']
        rY = video['resolutionY']
        resolution = f'{rX}x{rY}'
        bitrate = video['bitrate']
        colorfulness = video['colorfulness']
        fps = video['fps']
        return f'FPS = {fps}\nResolution = {resolution}\nBitrate = {bitrate}\nColorfulness = {colorfulness}\nSI = {SI}\nTI = {TI}'
    def calc(name, dist_videos, path = None, info = None):
        global firstTime, metrics
        metrics_vectors = [[0. for j in range(len(dist_videos))] for i in range(len(dist_videos[0]['metrics']) - 1)]
        ss_vector = [0. for j in range(len(dist_videos))]
        cor_arr = [0. for j in range(len(dist_videos[0]['metrics']) - 1)]
        for i in range(len(dist_videos)):
            vid = dist_videos[i]
            j = 0
            for metr, mean in vid['metrics'].items():
                if metr == 'subjective_score':
                    ss_vector[i] = mean
                    continue
                if metr in reverse_metrics:
                    metrics_vectors[j][i] = -mean
                else:
                    metrics_vectors[j][i] = mean
                if firstTime:
                    metrics.append(metr)
                j += 1
            firstTime = False
        ss_vector =  np.array(ss_vector)
        for i in range(len(metrics)):
            metrics_vectors[i] = np.array(metrics_vectors[i])
            for j in range(len(metrics_vectors[i])):
                if np.isnan(metrics_vectors[i][j]):
                    metrics_vectors[i][j] = 0.
            if cor_type == "spearman":
                cor_arr[i] = scipy.stats.spearmanr(ss_vector, metrics_vectors[i]).correlation
            elif cor_type == "kendall":
                cor_arr[i] = scipy.stats.kendalltau(ss_vector, metrics_vectors[i])[0]
            else:
                cor_arr[i] = scipy.stats.pearsonr(ss_vector, metrics_vectors[i]).statistic
        return np.array(cor_arr)
    
    group = video['group']
    cor_arr = np.array([])
    info = get_info(video)
    if video['dataset'] == 'SR+Codecs':
        cnt = 0
        for codec, dist_videos in video['dist_videos'].items():
            cnt += 1
            if cor_arr.size == 0:
                cor_arr = calc(f'Name: \"{group}\" Codec: \"{codec}\"', dist_videos, path + f'___Codec={codec}___Correlation={cor_type}.png', info)
            else:
                cor_arr += calc(f'Name: \"{group}\" Codec: \"{codec}\"', dist_videos, path + f'___Codec={codec}___Correlation={cor_type}.png', info)
        cor_arr /= cnt
    else:
        cor_arr = calc(f'Name : \"{group}\"', video['dist_videos'], path + f'___Correlation={cor_type}.png', info)
    return np.array(cor_arr)      
        
with open(stats_path, 'r') as f:
    full = json.load(f)
corp = np.array([])
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cor1p = np.array([])
cor2p = np.array([])
cor3p = np.array([])
cor4p = np.array([])
corpCI = np.array([])
cor1pCI = np.array([])
cor2pCI = np.array([])
cor3pCI = np.array([])
cor4pCI = np.array([])

cnt = 0
for video in full:
    cnt += 1
    if cnt > mxcnt:
        break
    name = video['group']
    res = get_correlation(video, os.path.join(pearson_path, f'Name={name}'), 'pearson')
    if corpCI.size == 0:
        for i in range(res.shape[0]):
            corpCI = np.append(corpCI, [res[i]])
        corpCI = corpCI.reshape(corpCI.shape[0], 1)
    else:
        tmparr = np.array([])
        for i in range(res.shape[0]):
            tmparr = np.append(tmparr, np.append(corpCI[i], res[i]))
        tmparr = tmparr.reshape(corpCI.shape[0], tmparr.shape[0] // corpCI.shape[0])
        corpCI = tmparr
    if video['dataset'] == 'SR Dataset':
        cnt1 += 1
        if cor1p.size == 0:
            cor1p = np.copy(res)
        else:
            cor1p += np.copy(res)
        if cor1pCI.size == 0:
            for i in range(res.shape[0]):
                cor1pCI = np.append(cor1pCI, [res[i]])
            cor1pCI = cor1pCI.reshape(cor1pCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor1pCI[i], res[i]))
            tmparr = tmparr.reshape(cor1pCI.shape[0], tmparr.shape[0] // cor1pCI.shape[0])
            cor1pCI = tmparr
    elif video['dataset'] == 'SR+Codecs':
        cnt2 += 1
        if cor2p.size == 0:
            cor2p = np.copy(res)
        else:
            cor2p += np.copy(res)
        if cor2pCI.size == 0:
            for i in range(res.shape[0]):
                cor2pCI = np.append(cor2pCI, [res[i]])
            cor2pCI = cor2pCI.reshape(cor2pCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor2pCI[i], res[i]))
            tmparr = tmparr.reshape(cor2pCI.shape[0], tmparr.shape[0] // cor2pCI.shape[0])
            cor2pCI = tmparr
    elif video['dataset'] == 'VSR_Benchmark':
        cnt3 += 1
        if cor3p.size == 0:
            cor3p = np.copy(res)
        else:
            cor3p += np.copy(res)
        if cor3pCI.size == 0:
            for i in range(res.shape[0]):
                cor3pCI = np.append(cor3pCI, [res[i]])
            cor3pCI = cor3pCI.reshape(cor3pCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor3pCI[i], res[i]))
            tmparr = tmparr.reshape(cor3pCI.shape[0], tmparr.shape[0] // cor3pCI.shape[0])
            cor3pCI = tmparr
    elif video['dataset'] == 'VUB_Benchmark':
        cnt4 += 1
        if cor4p.size == 0:
            cor4p = np.copy(res)
        else:
            cor4p += np.copy(res)
        if cor4pCI.size == 0:
            for i in range(res.shape[0]):
                cor4pCI = np.append(cor4pCI, [res[i]])
            cor4pCI = cor4pCI.reshape(cor4pCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor4pCI[i], res[i]))
            tmparr = tmparr.reshape(cor4pCI.shape[0], tmparr.shape[0] // cor4pCI.shape[0])
            cor4pCI = tmparr           
cor1p /= cnt1
cor2p /= cnt2
cor3p /= cnt3
cor4p /= cnt4
corp = (cor1p + cor2p + cor3p + cor4p) / 4

cors = np.array([])
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cor1s = np.array([])
cor2s = np.array([])
cor3s = np.array([])
cor4s = np.array([])
corsCI = np.array([])
cor1sCI = np.array([])
cor2sCI = np.array([])
cor3sCI = np.array([])
cor4sCI = np.array([])

cnt = 0
for video in full:
    cnt += 1
    if cnt > mxcnt:
        break
    name = video['group']
    res = get_correlation(video, os.path.join(spearman_path, f'Name={name}'), "spearman")
    if corsCI.size == 0:
        for i in range(res.shape[0]):
            corsCI = np.append(corsCI, [res[i]])
        corsCI = corsCI.reshape(corsCI.shape[0], 1)
    else:
        tmparr = np.array([])
        for i in range(res.shape[0]):
            tmparr = np.append(tmparr, np.append(corsCI[i], res[i]))
        tmparr = tmparr.reshape(corsCI.shape[0], tmparr.shape[0] // corsCI.shape[0])
        corsCI = tmparr
    if video['dataset'] == 'SR Dataset':
        cnt1 += 1
        if cor1s.size == 0:
            cor1s = np.copy(res)
        else:
            cor1s += np.copy(res)
        if cor1sCI.size == 0:
            for i in range(res.shape[0]):
                cor1sCI = np.append(cor1sCI, [res[i]])
            cor1sCI = cor1sCI.reshape(cor1sCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor1sCI[i], res[i]))
            tmparr = tmparr.reshape(cor1sCI.shape[0], tmparr.shape[0] // cor1sCI.shape[0])
            cor1sCI = tmparr
    elif video['dataset'] == 'SR+Codecs':
        cnt2 += 1
        if cor2s.size == 0:
            cor2s = np.copy(res)
        else:
            cor2s += np.copy(res)
        if cor2sCI.size == 0:
            for i in range(res.shape[0]):
                cor2sCI = np.append(cor2sCI, [res[i]])
            cor2sCI = cor2sCI.reshape(cor2sCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor2sCI[i], res[i]))
            tmparr = tmparr.reshape(cor2sCI.shape[0], tmparr.shape[0] // cor2sCI.shape[0])
            cor2sCI = tmparr
    elif video['dataset'] == 'VSR_Benchmark':
        cnt3 += 1
        if cor3s.size == 0:
            cor3s = np.copy(res)
        else:
            cor3s += np.copy(res)
        if cor3sCI.size == 0:
            for i in range(res.shape[0]):
                cor3sCI = np.append(cor3sCI, [res[i]])
            cor3sCI = cor3sCI.reshape(cor3sCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor3sCI[i], res[i]))
            tmparr = tmparr.reshape(cor3sCI.shape[0], tmparr.shape[0] // cor3sCI.shape[0])
            cor3sCI = tmparr
    elif video['dataset'] == 'VUB_Benchmark':
        cnt4 += 1
        if cor4s.size == 0:
            cor4s = np.copy(res)
        else:
            cor4s += np.copy(res)
        if cor4sCI.size == 0:
            for i in range(res.shape[0]):
                cor4sCI = np.append(cor4sCI, [res[i]])
            cor4sCI = cor4sCI.reshape(cor4sCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor4sCI[i], res[i]))
            tmparr = tmparr.reshape(cor4sCI.shape[0], tmparr.shape[0] // cor4sCI.shape[0])
            cor4sCI = tmparr
cor1s /= cnt1
cor2s /= cnt2
cor3s /= cnt3
cor4s /= cnt4
cors = (cor1s + cor2s + cor3s + cor4s) / 4

cork = np.array([])
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cor1k = np.array([])
cor2k = np.array([])
cor3k = np.array([])
cor4k = np.array([])
corkCI = np.array([])
cor1kCI = np.array([])
cor2kCI = np.array([])
cor3kCI = np.array([])
cor4kCI = np.array([])

cnt = 0
for video in full:
    cnt += 1
    if cnt > mxcnt:
        break
    name = video['group']
    res = get_correlation(video, os.path.join(spearman_path, f'Name={name}'), "kendall")
    if corkCI.size == 0:
        for i in range(res.shape[0]):
            corkCI = np.append(corkCI, [res[i]])
        corkCI = corkCI.reshape(corkCI.shape[0], 1)
    else:
        tmparr = np.array([])
        for i in range(res.shape[0]):
            tmparr = np.append(tmparr, np.append(corkCI[i], res[i]))
        tmparr = tmparr.reshape(corkCI.shape[0], tmparr.shape[0] // corkCI.shape[0])
        corkCI = tmparr
    if video['dataset'] == 'SR Dataset':
        cnt1 += 1
        if cor1k.size == 0:
            cor1k = np.copy(res)
        else:
            cor1k += np.copy(res)
        if cor1kCI.size == 0:
            for i in range(res.shape[0]):
                cor1kCI = np.append(cor1kCI, [res[i]])
            cor1kCI = cor1kCI.reshape(cor1kCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor1kCI[i], res[i]))
            tmparr = tmparr.reshape(cor1kCI.shape[0], tmparr.shape[0] // cor1kCI.shape[0])
            cor1kCI = tmparr
    elif video['dataset'] == 'SR+Codecs':
        cnt2 += 1
        if cor2k.size == 0:
            cor2k = np.copy(res)
        else:
            cor2k += np.copy(res)
        if cor2kCI.size == 0:
            for i in range(res.shape[0]):
                cor2kCI = np.append(cor2kCI, [res[i]])
            cor2kCI = cor2kCI.reshape(cor2kCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor2kCI[i], res[i]))
            tmparr = tmparr.reshape(cor2kCI.shape[0], tmparr.shape[0] // cor2kCI.shape[0])
            cor2kCI = tmparr
    elif video['dataset'] == 'VSR_Benchmark':
        cnt3 += 1
        if cor3k.size == 0:
            cor3k = np.copy(res)
        else:
            cor3k += np.copy(res)
        if cor3kCI.size == 0:
            for i in range(res.shape[0]):
                cor3kCI = np.append(cor3kCI, [res[i]])
            cor3kCI = cor3kCI.reshape(cor3kCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor3kCI[i], res[i]))
            tmparr = tmparr.reshape(cor3kCI.shape[0], tmparr.shape[0] // cor3kCI.shape[0])
            cor3kCI = tmparr
    elif video['dataset'] == 'VUB_Benchmark':
        cnt4 += 1
        if cor4k.size == 0:
            cor4k = np.copy(res)
        else:
            cor4k += np.copy(res)
        if cor4kCI.size == 0:
            for i in range(res.shape[0]):
                cor4kCI = np.append(cor4kCI, [res[i]])
            cor4kCI = cor4kCI.reshape(cor4kCI.shape[0], 1)
        else:
            tmparr = np.array([])
            for i in range(res.shape[0]):
                tmparr = np.append(tmparr, np.append(cor4kCI[i], res[i]))
            tmparr = tmparr.reshape(cor4kCI.shape[0], tmparr.shape[0] // cor4kCI.shape[0])
            cor4kCI = tmparr
cor1k /= cnt1
cor2k /= cnt2
cor3k /= cnt3
cor4k /= cnt4
cork = (cor1k + cor2k + cor3k + cor4k) / 4

c1 = cnt1
c2 = cnt2
c3 = cnt3
c4 = cnt4
cgl = c1 + c2 + c3 + c4
CIp = [weigh_function(corpCI[i], 0) for i in range(len(corpCI))]
CI1p = [weigh_function(cor1pCI[i], 0) for i in range(len(cor1pCI))]
CI2p = [weigh_function(cor2pCI[i], 0) for i in range(len(cor2pCI))]
CI3p = [weigh_function(cor3pCI[i], 0) for i in range(len(cor3pCI))]
CI4p = [weigh_function(cor4pCI[i], 0) for i in range(len(cor4pCI))]
lensp = [(CI1p[i][1] + CI2p[i][1] + CI3p[i][1] + CI4p[i][1]) / 8 - (CI1p[i][0] + CI2p[i][0] + CI3p[i][0] + CI4p[i][0]) / 8 for i in range(len(CI1p))]
CIp = [[max(corp[i] - lensp[i], 0), min(corp[i] + lensp[i], 1)] for i in range(len(lensp))]


CIs = [weigh_function(corsCI[i], 0) for i in range(len(corsCI))]
CI1s = [weigh_function(cor1sCI[i], 0) for i in range(len(cor1sCI))]
CI2s = [weigh_function(cor2sCI[i], 0) for i in range(len(cor2sCI))]
CI3s = [weigh_function(cor3sCI[i], 0) for i in range(len(cor3sCI))]
CI4s = [weigh_function(cor4sCI[i], 0) for i in range(len(cor4sCI))]
lenss = [(CI1s[i][1] + CI2s[i][1] + CI3s[i][1] + CI4s[i][1]) / 8 - (CI1s[i][0] + CI2s[i][0] + CI3s[i][0] + CI4s[i][0]) / 8 for i in range(len(CI1s))]
CIs = [[max(cors[i] - lenss[i], 0), min(cors[i] + lenss[i], 1)] for i in range(len(lenss))]

CIk = [weigh_function(corkCI[i], 0) for i in range(len(corkCI))]
CI1k = [weigh_function(cor1kCI[i], 0) for i in range(len(cor1kCI))]
CI2k = [weigh_function(cor2kCI[i], 0) for i in range(len(cor2kCI))]
CI3k = [weigh_function(cor3kCI[i], 0) for i in range(len(cor3kCI))]
CI4k = [weigh_function(cor4kCI[i], 0) for i in range(len(cor4kCI))]
lensk = [(CI1k[i][1] + CI2k[i][1] + CI3k[i][1] + CI4k[i][1]) / 8 - (CI1k[i][0] + CI2k[i][0] + CI3k[i][0] + CI4k[i][0]) / 8 for i in range(len(CI1k))]
CIk = [[max(cork[i] - lensk[i], 0), min(cork[i] + lensk[i], 1)] for i in range(len(lensk))]
mn = 0.
hcplot("Global", corp, cors, cork, CIp, CIs, CIk, mn, 1., os.path.join(img_path, f'Global_CI'))
hcplot("SR Dataset", cor1p, cor1s, cor1k, CI1p, CI1s, CI1k, mn, 1., os.path.join(img_path, f'SR_Dataset_CI'))
hcplot("SR+Codecs", cor2p, cor2s, cor2k, CI2p, CI2s, CI2k, mn, 1., os.path.join(img_path, f'SR+Codecs_CI'))
hcplot("VSR Benchmark", cor3p, cor3s, cor3k, CI3p, CI3s, CI3k, mn, 1., os.path.join(img_path, f'VSR_Benchmark_CI'))
hcplot("VUB Benchmark", cor4p, cor4s, cor4k, CI4p, CI4s, CI4k, mn, 1., os.path.join(img_path, f'VUB_Benchmark_CI'))
with open(corstats_path, 'r') as f:
    resdict = json.load(f)
dss = ["Global", "SR Dataset", "SR+Codecs", "VSR Benchmark", "VUB Benchmark"]
for mname in resdict.keys():
    for ds in dss:
        if not ds in resdict[mname].keys():
            resdict[mname][ds] = { "Pearson" : -1, "Spearman" : -1, "Kendall" : -1}
with open(corstats_path, 'w') as f:
    json.dump(resdict, f, sort_keys = True, indent = 4)

with open(CIstats_path, 'r') as f:
    CIresdict = json.load(f)
for mname in CIresdict.keys():
    for ds in dss:
        if not ds in CIresdict[mname].keys():
            CIresdict[mname][ds] = { "Pearson" : [-1, -1], "Spearman" : [-1, -1], "Kendall" : [-1, -1]}
with open(CIstats_path, 'w') as f:
    json.dump(CIresdict, f, sort_keys = True, indent = 4)
