wget https://f000.backblazeb2.com/file/jamesapohadi-dataset/Resnet50_Final.pth
mkdir weights
mv Resnet50_Final.pth weights/
wget https://f000.backblazeb2.com/file/jamesapohadi-dataset/widerface.zip
unzip -qq widerface.zip
mv widerface data/

python3 -m pip install python-telegram-bot