This is the project for CSCE689-Advanced Topics of Deep Learning.
The objective of this project is to identify human limping.

For now, the model is trained based on Walking Gait Dataset which could be found at http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/

In this version, the model use is /models/front-slow/model.h5
test.py will show the result in test dataset.

video prediction:
put the video in npy format, json file in a directory:

video
|--<video in npy format>
|--output
   |--<JSON file>


Then change the path in main function in video-pred.py, run script.
It will generate the prediction figure, time label json file and a animation which shows the video, skeleton and prediction result.
