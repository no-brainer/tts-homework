# TTS project

## Installation

```shell script
pip3 install -r requirements.txt

mkdir -p ./data/datasets
# vocoder
git clone https://github.com/NVIDIA/waveglow.git
gdown --id 1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF -O ./data/waveglow_256channels_universal_v5.pt

# alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip -O ./data/alignments.zip
unzip ./data/alignments.zip -d ./data

# dataset
cd ./data/datasets
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
cd ../..
```
