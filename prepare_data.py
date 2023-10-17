from pathlib import Path
from glob import glob
import os
import re
import pickle

dir = '../../OASIS-1/'
dir = Path(dir)

pattern = str(dir)+'/OAS1_*_MR*/PROCESSED/MPRAGE/T88_111/OAS1_*_MR*_mpr_n*_anon_111_t88_masked_gfc_MNI152_T1_1mm.nii.gz'
files = glob(pattern)

print(len(files))

from mfmri.mfmri.mfmri_lite import LiteMFractalMRI

mfmri = LiteMFractalMRI()

DATA_FILE = 'hurst_data.pickle'
if not os.path.exists(DATA_FILE):
    data = {}
else:
    with open(DATA_FILE,'rb') as fp:
        data = pickle.load(fp)

for i,file in enumerate(files):
    file_id = re.findall('(OAS1_.{4}_MR.)_',file)[0]
    if file_id not in data.keys():
        print(f'{i}/{len(files)}: {file_id}')
        mfmri.pipeline(file)
        data[file_id] = mfmri.get_hurst()
    else:
        print(f'{i}/{len(files)}: {file_id} exists')
    with open(DATA_FILE,'wb') as fp:
        pickle.dump(data,fp)
        
