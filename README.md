
### Installation:
```bash

# install pip and virtualenv if necessary
curl -O https://raw.github.com/pypa/pip/master/contrib/get-pip.py
python get-pip.py
pip install virtualenv

# clone the repo and install dependencies
git clone https://github.com/StanfordVisionSystems/vfeedbacknet
cd vfeedbacknet
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# install the vfeedbacknet library
python setup.py install

# download dataset (if training from scratch)
pip install twentybn-dl
twentybn-dl obtain jester
  Set storage directory [Enter for "/home/user/20bn-datasets"]:
  Using: '/home/user/20bn-datasets' as storage.
  Will now get chunks for: 'jester'
  Will now download chunks.
  Downloading: 'https://s3-eu-west-1.amazonaws.com/20bn-public-datasets/jester/v1/20bn-jester-v1-00'
  ...
```

### Usage (code located in `scripts` directory):
```
example instruction
```

### Inference using pretrained model:
```
example instruction
```

### Training the model yourself:
```
example instruction
```

### About Me:

### Acknowledgements:

### Citation:
```
```