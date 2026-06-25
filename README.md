# TriDETR: An Effective End-to-End Object Detector

This is the official implementation of **TriDETR: An effective end-to-end object detector**. 

This repository is built upon the official [RT-DETR](https://github.com/lyuwenyu/RT-DETR) implementation (specifically the `main` branch). 

The code for our proposed TriDETR method is located in the **`rau-trifocal-fix`** branch.

---

## 🛠️ Installation

### Requirements
- **Python:** 3.11

### Setup Environment
1. Clone this repository and checkout the proposed method branch:
```bash
git clone https://github.com/hudzg/my-rt-detr.git
cd my-rt-detr/rtdetr_pytorch
git checkout rau-trifocal-fix
```
Install the required dependencies:

```bash
pip install -r requirements.txt
pip install numpy==1.26.4
```

### Training
To train the model, run the following command:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_voc.yml --amp --seed 1706
```

### Inference / Evaluation
To evaluate or run inference using a trained model checkpoint, use the --test-only flag.

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_voc.yml -r output/rtdetr_r18vd_6x_voc/checkpoint0071.pth --test-only
```

### Acknowledgements
This project is based on the excellent work from [RT-DETR](https://github.com/lyuwenyu/RT-DETR). We sincerely thank the authors for open-sourcing their code.
