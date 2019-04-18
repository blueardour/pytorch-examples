
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch resnet18 -b 128 --iter-size 2 --case 'iter-size2_batch128'
cd -
