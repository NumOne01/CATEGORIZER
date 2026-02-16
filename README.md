## Requirements
- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm

## Datasets
1. Download four public benchmarks for fine-grained dataset
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - [MIT-67: Indoor Scene Recognition](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar)
   - [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar)
   - [FGVC-Aircraft Benchmark](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

2. Extract the tgz or zip file into `./data/` (Exceptionally, for CUB-200-2011, put the files in a `./data/CUB200`)

##
This repository contains a modified version of vast repository.
First install the dependencies using:
```
pip install -r requirements.txt
```
Then 
```
pip install -e .
```


## Train Examples
```
python train.py \
--model resnet18 \
--dataset mit \
--alpha 32 \
--mrg 0.1 \
--lr 1e-4 \
--warm 5 \
--epochs 60 \
--batch-size 120 \
```

## Acknowledgements
This repository is heavily built based on the following repository:

- [Proxy Anchor-based Unsupervised Learning for Continuous Generalized Category Discovery](https://github.com/Hy2MK/CGCD)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
