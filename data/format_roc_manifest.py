import xml.etree.ElementTree as ET
import glob, os, csv, random
from collections import Counter

xml_path = r'data/roc/annotations-consensus-ma-only.xml'
img_dir = r'data/roc'
manifest_path = r'data/manifests/roc_manifest.csv'

tree = ET.parse(xml_path)
root = tree.getroot()

ma_counts = {}
for item in root.findall('annotations-per-image'):
    name = item.get('imagename')
    annotations = item.findall('annotation')
    ma_counts[name] = len(annotations)

random.seed(42)
imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

grade_buckets = {0: [], 1: [], 2: [], 3: [], 4: []}
for fname in imgs:
    cnt = ma_counts.get(fname, 0)
    if cnt == 0: grade = 0
    elif cnt <= 5: grade = 1
    elif cnt <= 15: grade = 2
    elif cnt <= 25: grade = 3
    else: grade = 4
    grade_buckets[grade].append(fname)

split_map = {}
for g, flist in grade_buckets.items():
    random.shuffle(flist)
    n = len(flist)
    n_test = max(1, int(round(n * 0.2)))
    n_val = max(1, int(round(n * 0.16)))
    if n <= 3:
        n_test = 1
        n_val = 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_val > 0: n_val -= 1
        else: n_test -= 1
        
    for i, fname in enumerate(flist):
        if i < n_train:
            split_map[fname] = 'train'
        elif i < n_train + n_val:
            split_map[fname] = 'val'
        else:
            split_map[fname] = 'test'

rows = []
for fname in imgs:
    cnt = ma_counts.get(fname, 0)
    if cnt == 0: grade = 0
    elif cnt <= 5: grade = 1
    elif cnt <= 15: grade = 2
    elif cnt <= 25: grade = 3
    else: grade = 4
    
    pat_id = 'PAT_ROC_' + fname.split('_')[0].replace('image', '')
    rel_path = f'data/roc/{fname}'
    rows.append({
        'path': rel_path,
        'label': grade,
        'dataset': 'roc_iowa',
        'patient_id': pat_id,
        'eye': 'unknown',
        'split': split_map[fname],
        'is_ungradable': 0
    })

os.makedirs('data/manifests', exist_ok=True)
with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'label', 'dataset', 'patient_id', 'eye', 'split', 'is_ungradable'])
    writer.writeheader()
    writer.writerows(rows)

print('Updated roc_manifest.csv successfully!')
print('Split counts:', Counter([r['split'] for r in rows]))
print('Label counts:', Counter([r['label'] for r in rows]))
