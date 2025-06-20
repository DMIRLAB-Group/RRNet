
for id in {0,1,2,3,4} ; do
    CUDA_VISIBLE_DEVICES=0 python main.py --model=Mine --dataset=BC_hete_uncon_decom --expID=$id &
    CUDA_VISIBLE_DEVICES=1 python main.py --model=Mine --dataset=Flickr_hete_uncon_decom --expID=$id &
    CUDA_VISIBLE_DEVICES=2 python main.py --model=Mine --dataset=Flickr_uncon_decom --expID=$id &
    CUDA_VISIBLE_DEVICES=3 python main.py --model=Mine --dataset=BC_uncon_decom --expID=$id &
done
wait