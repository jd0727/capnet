# CapNet: Learning Insulator Self-Blast from Bounding Box 
## Data preparation
We recommend users to organize data in VOC format
``` 
-root
    -JPRGImages
        -000001.jpg
        -...
    -Annotations
        -000001.xml
        -...
    -ImageSets
        -Main
            -train.txt
            -test.txt
            -...
```
Run the python script to split the insulator. The output annotation file name, hyper parameters, and the custom setting are configured in the same python script.
```
python scripts/_split_insu_.py
```
## Train CapNet
Run _wpsv_.py for CapNet training
```
python scripts/_wpsv_.py
```
## Result Visualize
Run _wpsv_visual_.py for model visualize
```
python scripts/_wpsv_visual_.py
```
