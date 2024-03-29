
# Recognition-System

Welcome to our Recognition System, a comprehensive solution for face recognition and face detection powered by state-of-the-art models such as ArcFace, SCRFD. This system integrates cutting-edge technology to provide accurate and efficient recognition capabilities for various applications.
## Authors

- [@Khuy](https://github.com/HyUvscode)



## Appendix

Any additional information goes here


## Run Locally

Clone the project

```bash
  git clone https://github.com/HyUvscode/Recognition-System.git
```

Go to the project directory

```bash
  cd Recognition-System
```

Install conda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Create env with python 3.9
```bash
  conda create -n face_recognize_system python=3.9
```


Install dependencies

```bash
  pip install -r requirements.txt
```

DATA FORMAT
```bash
_datasets
____backup
____data
____face_features
____new_persons
_________person_1
__________________image1.png
__________________image2.png
_________person_2
__________________image1.png
__________________image2.png
```

Start add persons to system

```bash
  python3 detect.py
```

Run recognize file

```bash
    python3 recognize.py
```

## Support

For support, email khanhuy0915@gmail.com or join our Slack channel.

