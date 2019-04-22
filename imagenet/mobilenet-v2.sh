
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch mobilenet-v2 --epochs 120 -b 128 --iter-size 1 --lr 0.045 --lr_policy 'decay' --lr_decay 0.95 --case 'iter-size1_batch128' --tensorboard
cd -
