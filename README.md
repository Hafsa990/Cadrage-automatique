To try the model by yourself, you will first need to make a new virtual environment with python:
`python3 -m venv env`

After you make a new environment, you should be able to navigate to it and activate it with this code:
`source env/bin/activate`

Once this is configurted, you can download the requirements with pip:
`pip install -r requirements.txt`

The utility functions are configured in file `utils.py`. These can be used anywhere as long as the packages are installed.
There is an example jupyter notebook `auto_framing.ipynb` which shows how to use the functions.

Please note that you will need the following files/folders:

1. models/<name_of_your_unet_model.pth>: This will contain your trained U-net model. You can either use the one provided in the repository or bring your own.
2. video_samples/<example_video.mp4>: You will need to add example videos at the root of the folder.