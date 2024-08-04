import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import re


# ******** Helper functions for plotting results ********
# ********************************************************
def display_pair(image, mask, text=False):
    if torch.is_tensor(image):
        image = image.permute(1,2,0).numpy()
        mask = mask.permute(1,2,0).numpy()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.subplots_adjust(wspace=0, hspace=0)
    for ax in axes.flat:
        ax.axis("off")    
        ax.tick_params(labelleft=False, labelbottom=False)
    axes[0].imshow(image)
    axes[1].imshow(mask)
    if text:
        axes[0].text(0, 15, "Masque de référence", c="red", fontsize=20)
        axes[1].text(0, 15, "Masque prédit", c="red", fontsize=20)
    plt.show()


def display_model_comparisons(real_image,image, maskmodel1, maskmodel2, text=True):
    if torch.is_tensor(image):
        real_image = real_image.permute(1, 2, 0).numpy()
        image = image.permute(1,2,0).numpy()
        maskmodel1 = maskmodel1.permute(1,2,0).numpy()
        maskmodel2 = maskmodel2.permute(1,2,0).numpy()
    fig, axes = plt.subplots(2,2, figsize=(15,10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for ax in axes.flat:
        ax.axis("off")    
        ax.tick_params(labelleft=False, labelbottom=False)
    axes[0][0].imshow(real_image)
    axes[0][1].imshow(image)
    axes[1][0].imshow(maskmodel1)
    axes[1][1].imshow(maskmodel2)

    if text:
        axes[0][0].text(0, 15, "Image de référence", c="red", fontsize=15)
        axes[0][1].text(0, 15, "Masque de référence", c="red", fontsize=15)
        axes[1][0].text(0, 15, "A: Masque prédit", c="red", fontsize=15)
        axes[1][1].text(0, 15, "B: Masque prédit", c="red", fontsize=15)

    plt.show()


def display_threes(real_image,image, maskmodel, text=True):
    if torch.is_tensor(image):
        real_image = real_image.permute(1, 2, 0).numpy()
        image = image.permute(1,2,0).numpy()
        maskmodel = maskmodel.permute(1,2,0).numpy()
    fig, axes = plt.subplots(1,3, figsize=(10,5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for ax in axes.flat:
        ax.axis("off")    
        ax.tick_params(labelleft=False, labelbottom=False)
    axes[0].imshow(real_image)
    axes[1].imshow(image)
    axes[2].imshow(maskmodel)

    if text:
        axes[0].text(0, 15, "Image de référence", c="red", fontsize=15)
        axes[1].text(0, 15, "Masque de référence", c="red", fontsize=15)
        axes[2].text(0, 15, "Masque prédit", c="red", fontsize=15)

    return fig
# ********************************************************





# ******** autoframing ********
def transform_to_input(img_path: str):
    """    
    Helper function to change the sizing 
    of the images so it can be passed to the Unet model

    Args:
        img_path (str): folder path to image to transfrom - compatible with U-net model
    """
    IMG_SIZE = 256
    tranform_funcs = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ])
    
    image = Image.open(img_path)
    return tranform_funcs(image)


def atoi(text):
    """
    Helper function Get int if exists
    Args:
        text (str): text to convert to int
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """    
    Helper function
    alist.sort(key=natural_keys) sorts in human order
    Args:
        text (str): text to sort
    """
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def find_largest_width_height(frames_path):
    """
    Finds the largest width and height in a folder of frames

    Args:
        frames_path (str): folder path to the frames
    """
    frames = [img for img in os.listdir(frames_path) if img.endswith("_autoframe.jpg")]
    width = 0
    height = 0
    for frame in frames:
        img = cv2.imread((os.path.join(frames_path, frame)))
        if img.shape[0] > width:
            width = img.shape[0]
        if img.shape[1] > height:
            height = img.shape[1]
    return width,height


def generate_frames(video_path:str,frames_path:str):
    """
    Generates frames for the specified filename

    Args:
        video_path (str): folder path to the video
        frames_path (str): folder path to the frames

    Returns:
        fps (float): frames per second of the input video
    """
    interval = 1
    frame_count = 0
    if not(os.path.isdir(frames_path)):
        os.makedirs(frames_path)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while True:
        # Read the current frame
        ret, frame = video.read()
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count + interval)
        frame_count += interval
        name = f"{frames_path}/{frame_count}.jpg"
        if ret:
            cv2.imwrite(name, frame)
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    return fps


def run_model_over_frames(frames_path:str,masks_path: str,model_path:str):
    """    
    Function that will load the Unet model,
    will run the Unet model over every frame that is generated in frames_path directory.
    It will produce a mask image for every frame and store it in masks_path

    Args:
        frames_path (str): folder path to the frames
        masks_path (str): folder path to the masks
        model_path (str): folder path to the model
    """
    frames = sorted(glob.glob(f"{frames_path}/*"))
    model = torch.load(model_path)

    if not(os.path.isdir(masks_path)):
        os.makedirs(masks_path)

    for frame in frames:
        input = transform_to_input(img_path = frame)
        filename = frame.split("/")[-1]
        output = model(input.unsqueeze(0))

        #sigmoid function to get 0 and 1 pixels
        output = F.sigmoid(output)
        output = (output>0.5).float()

        ##Save the output mask
        # Scale the pixel values to the range [0, 255]
        image_array = output[0].permute(1, 2, 0).numpy()
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        image_array = np.squeeze(image_array, axis=-1)
        image = Image.fromarray(image_array)
        # Save the image
        image.save(f"{masks_path}/{filename}_mask.png")
    print("generated masks")



def auto_frame_all(masks_path:str, frames_path:str, desired_width: int = None, desired_height: int = None, head_ratio:float = 0.0):
    """ 
    This function applies the auto-framing algorithm
    It reframes a mask image, and then applies the reference points to 
    the sister frame

    Args:
        masks_path (str): folder path to the masks
        frames_path (str): folder path to the frames
        desired_width (int, optional): desired height to frame the final frames with
        desired_height (int, optional): desired width to frame the final frames with
        head_ratio (float, optional): applies the head ratio to create a head frame, generally a 0.5(50%) is acceptable.
    """
    # Find the coordinates of the bounding box
    all_masks = sorted(glob.glob(f"{masks_path}/*"))
    
    for mask in all_masks:
        try:
            mask_array = np.array(Image.open(mask))
            rows = np.any(mask_array, axis=1)
            cols = np.any(mask_array, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
        except IndexError:
            ymin, ymax = 0 , len(rows)
            xmin, xmax = 0 , len(cols)

        # Adjust the bounding box to focus on the head
        if head_ratio > 0.0:
            height = ymax - ymin + 1
            head_height = int(height * head_ratio)
            head_ymax = ymin + head_height
        # Crop the image to the adjusted bounding box
            cropped_image = mask_array[ymin:head_ymax, xmin:xmax+1]
        else:
            cropped_image = mask_array[ymin:ymax+1, xmin:xmax+1]

        if desired_width is not None and desired_height is not None:
            # Resize the cropped image to desired dimensions
            cropped_image_pil = Image.fromarray((cropped_image * 255).astype(np.uint8))
            resized_image = cropped_image_pil.resize((desired_width, desired_height), Image.LANCZOS)
            final_image_arr = np.array(resized_image) / 255  # Convert back to 0-1 array
        else:
            final_image_arr = cropped_image
        # wirte image back to same directory ... 
        Image.fromarray(final_image_arr).save(mask.replace('_mask.png', '_mask_autoframe.png'))
            
        # Apply the dimensions to be cut on original images too 
        #This will generate dynamically the path of the original frame from the mask frame.
        img_filename  = os.path.basename(mask).replace("_mask.png","")
        original_frame_path = f"{frames_path}/{img_filename}"
        original_frame_array = np.array(Image.open(original_frame_path))

        scale_y = original_frame_array.shape[0]/mask_array.shape[0]
        scale_x  = original_frame_array.shape[1]/mask_array.shape[1]
        
        main_ymin = int(ymin * scale_y)
        main_ymax = int(head_ymax * scale_y) if head_ratio > 0.0 else int(ymax * scale_y)
        main_xmin = int(xmin * scale_x)
        main_xmax = int(xmax * scale_x)

        cropped_original_image = original_frame_array[main_ymin:main_ymax+1, main_xmin:main_xmax+1]

        cropped_image = Image.fromarray(cropped_original_image)
        if desired_height is not None and desired_width is not None:
            cropped_image.resize((720, 1280), Image.BICUBIC)
        
        new_path = original_frame_path.replace(".jpg","_autoframe.jpg")
        cropped_image.save(new_path)
    print("autoframed all")



def  make_new_video(frames_path:str, fps: float, new_video_path:str):
    """
    Makes an mp4 video by appending all the frames together
    Args:
        frames_path (str): folder path to the frames
        fps (float): frames per second of the video
        new_video_path (str): folder path to the new video generated
    """

    ## Set the width and height as one value. to be decided later
    width, height = find_largest_width_height(frames_path)
    video = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), round(fps), (width,height))
    # Loop through the images and add them to the video
    # Get only those that end with _autoframed.png, and sort them 
    frames = [img for img in os.listdir(frames_path) if img.endswith("_autoframe.jpg")]
    frames.sort(key=natural_keys)

    for frame in frames:
        img = cv2.resize(cv2.imread(os.path.join(frames_path, frame)), (width, height))
        video.write(img)
    # Release the VideoWriter object
    video.release()

    print(f"Video created: {new_video_path}")
# ********************************************************