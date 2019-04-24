
if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

cd $FASTDIR/git/pytorch-examples/imagenet
python main.py --data $FASTDIR/data/imagenet/ --arch 'mobilenet-v1' \
  --epochs 120 \
  --lr 0.045 --lr_policy 'decay' --lr_decay 0.95 --wd 4e-5 --nesterov \
  --case 'batch256-nesterov'
cd -
