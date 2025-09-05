
# bash tools/dist_test.sh  work_dir/gres/loss_weight/exist0.1_det0.1_ref1.0/20250510_224053/20250510_224053_exist0.1_det0.1_ref1.0.py 4 --load-from  work_dir/gres/loss_weight/exist0.1_det0.1_ref1.0/20250510_224053/latest.pth  --MTD_K 250
# bash tools/dist_test.sh  work_dir/gres/loss_weight/exist0.2_det0.1_ref0.5/20250511_013123/20250511_013123_exist0.2_det0.1_ref0.5.py 4 --load-from  work_dir/gres/loss_weight/exist0.2_det0.1_ref0.5/20250511_013123/latest.pth  --MTD_K 250
# bash tools/dist_test.sh  work_dir/gres/loss_weight/exist0.2_det0.05_ref1.0/20250511_042002/20250511_042002_exist0.2_det0.05_ref1.0.py 4 --load-from  work_dir/gres/loss_weight/exist0.2_det0.05_ref1.0/20250511_042002/latest.pth  --MTD_K 250
# bash tools/dist_test.sh  work_dir/gres/loss_weight/exist0.5_det0.1_ref1.0/20250511_071117/20250511_071117_exist0.5_det0.1_ref1.0.py 4 --load-from  work_dir/gres/loss_weight/exist0.5_det0.1_ref1.0/20250511_071117/latest.pth  --MTD_K 250
# bash tools/dist_test.sh  work_dir/gres/compare/all_ann/20250123_210009/20250123_210009_all_ann.py 4 --load-from  work_dir/gres/compare/all_ann/20250123_210009/latest.pth  --MTD_K 250

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh  configs/gres/PropVG-grefcoco.py 4 --load-from  work_dir/gres/PropVG-grefcoco.pth  --MTD_K 250 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh  configs/refcoco/PropVG-refcoco.py 4 --load-from  work_dir/refcoco/PropVG-refcoco.pth --MTD_K 100 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh  configs/refcoco+/PropVG-refcoco+.py 4 --load-from  work_dir/refcoco+/PropVG-refcoco+.pth --MTD_K 100 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh  configs/refcocog/PropVG-refcocog.py 1 --load-from  work_dir/refcocog/PropVG-refcocog.pth --MTD_K 100 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh  configs/refzom/PropVG-refzom.py 4 --load-from  work_dir/refzom/PropVG-refzom.pth --MTD_K 100 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh  configs/rrefcoco/PropVG-rrefcoco.py 4 --load-from  work_dir/rrefcoco/PropVG-rrefcoco.pth --MTD_K 100 --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh  configs/refcoco-mix/PropVG-refcoco-mix.py 1 --load-from  work_dir/refcoco-mix/PropVG-refcoco-mix.pth --MTD_K 100 --score-threshold 0.7
