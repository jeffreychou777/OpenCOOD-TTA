import sys
import runpy
import os

os.chdir('/home/zjf/DATACENTER2/data/code/OpenCOOD-TTA')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '

selfatt_args = 'python opencood/tools/train.py --hypes_yaml /home/zjf/DATACENTER2/data/code/OpenCOOD-TTA/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml'

selfatt_tta_args = 'python opencood/tools/tta.py --hypes_yaml /home/zjf/DATACENTER2/data/code/OpenCOOD-TTA/opencood/hypes_yaml/cptta/pointpillar_selfatt_contratta.yaml'

args = selfatt_tta_args

'python opencood/tools/tta.py --hypes_yaml /home/zjf/DATACENTER2/data/code/OpenCOOD-TTA/opencood/hypes_yaml/cptta/pointpillar_selfatt_contratta.yaml'

# args = 'python test.py 5 7'
args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
