
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch resnet18 -b 256 --iter-size 1 --case 'iter-size1_batch256'
cd -
