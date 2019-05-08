
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch resnet18 \
  --epochs 90 \
  --lr_fix_step 10 --lr_decay 0.3 --nesterov \
  --case 'batch256-nesterov' \
  -r
cd -
