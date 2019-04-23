
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch resnet18 --lr_decay 0.1 --case 'iter-size1_batch256' --tensorboard --nesterov
cd -
