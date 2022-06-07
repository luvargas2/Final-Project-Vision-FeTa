# Final-Project-Vision-FeTa
Computer Vision Final Proyect IBIO4490 - FeTA Challenge


In this project, we compare the 2D, 2.5D and 3D approaches for a multi-class fetal brain semantic segmentation task in the FeTa Dataset


### Feta Dataset

The Feta dataset - test is available in the next path: 
```
BCV003: /home/lfvargas10/Final-Project-Vision-FeTa/Data ## for ROG
BCV003: /home/lfvargas10/Final-Project-Vision-FeTa/Unet/data # for Unet 2D (patches) and Unet 2.5D
```


### Setup

Clone our repository with 
https://github.com/luvargas2/Final-Project-Vision-FeTa.git

You will find a feta.yml file to create and activate the proper virtual enviroment to run the testing and demo (main.py) of the main best models of each approach . 

To test the models run the following command, 

Inside the ROG folder
```
python main.py --test
```

Inside the Unet folder
```
python main.py 
```

All the arguments to customize your experiment are in the main.py of each folder (ROG/Unet)

### Setup
We add a demo that load our models trained weights and performs the segmentation of an specific patient of the test set. The script select the best model for each approach, and print the metrics on the terminal for all methods.

Inside the ROG folder
```
python main.py --test --mode demo
```


Inside the Unet folder
```
python main.py  --mode demo
```

Thank you :) 
