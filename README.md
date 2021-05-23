# BRAFT
## Requirements

```
conda create --name BRAFT
conda activate BRAFT
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```



## Demos

You can demo a trained model on a sequence of frames（Please adjust the script according to the specific method）

```
KITTI 2015
python demo.py --model=models/kitti/44block/raft-kitti.pth --path=demo-frames-kitti --method 44block --dataset kitti

MPI-sintel
python demo.py --model=models/sintel/44block/raft-sintel.pth --path=demo-frames-sintel --method 44block --dataset sintel
```



## Required Data

To evaluate/train BRAFT, you will need to download the required datasets.

- [Sintel](http://sintel.is.tue.mpg.de/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
```

## Evaluation

You can evaluate a trained model using `evaluate.py`

```
KITTI 2015
python evaluate.py --model=models/kitti/original/raft-kitti.pth --dataset=kitti --mixed_precision
python evaluate.py --model=models/kitti/4split/raft-kitti.pth --dataset=kitti --mixed_precision --method=4split
python evaluate.py --model=models/kitti/6split/raft-kitti.pth --dataset=kitti --mixed_precision --method=6split
python evaluate.py --model=models/kitti/8split/raft-kitti.pth --dataset=kitti --mixed_precision --method=8split
python evaluate.py --model=models/kitti/44block/raft-kitti.pth --dataset=kitti --mixed_precision --method=44block

MPI-Sintel
python evaluate.py --model=models/sintel/original/raft-sintel.pth --dataset=sintel --mixed_precision
python evaluate.py --model=models/sintel/4split/raft-sintel.pth --dataset=sintel --mixed_precision --method=4split
python evaluate.py --model=models/sintel/6split/raft-sintel.pth --dataset=sintel --mixed_precision --method=6split
python evaluate.py --model=models/sintel/8split/raft-sintel.pth --dataset=sintel --mixed_precision --method=8split
python evaluate.py --model=models/sintel/44block/raft-sintel.pth --dataset=sintel --mixed_precision --method=44block
```

## Training

Use single GPU for training（Please adjust the script according to the specific method）

```
KITTI2015
./train-kitti.sh
MPI-Sintel
./train-sintel.sh

```

