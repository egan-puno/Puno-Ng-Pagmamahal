import cv2
import numpy as np
import os

def soil(path):
    # Read the image
    image = cv2.imread(path)
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for soil color in HSV
    lower_soil = np.array([10, 50, 50]) 
    upper_soil = np.array([30, 255, 255]) 
    
    # Define the lower and upper bounds for grass color in HSV
    lower_grass = np.array([35, 40, 40]) 
    upper_grass = np.array([85, 255, 255]) 
    
    # Create masks for soil and grass
    soil_mask = cv2.inRange(hsv_image, lower_soil, upper_soil)
    grass_mask = cv2.inRange(hsv_image, lower_grass, upper_grass)
    
    # Invert the grass mask
    grass_mask_inv = cv2.bitwise_not(grass_mask)
    
    # Combine soil mask and inverted grass mask
    combined_mask = cv2.bitwise_and(soil_mask, grass_mask_inv)
    
    # Threshold the combined mask
    _, thresholded_mask = cv2.threshold(combined_mask, 49, 255, cv2.THRESH_BINARY)
    
    # Erode the mask to contract the regions
    kernel = np.ones((10, 10), np.uint8)
    contracted_mask = cv2.erode(thresholded_mask, kernel, iterations=1)
    
    # Create a colored image with soil regions highlighted
    colored_image_soil = np.zeros_like(image_rgb)
    colored_image_soil[contracted_mask > 0] = [255, 255, 0]
    
    # Convert the colored image back to BGR
    colored_image_soil = cv2.cvtColor(colored_image_soil, cv2.COLOR_BGR2RGB)
    
    return colored_image_soil

def land(path):
    # Read the image
    original_image = cv2.imread(path)
    
    # Make a copy of the original image
    copy_image = original_image.copy()
    
    # Convert the image from BGR to HSV
    copy_hsb = cv2.cvtColor(copy_image, cv2.COLOR_BGR2HSV)
    
    # Extract the blue channel
    blue_channel = copy_hsb[:, :, 0]
    
    # Decrease saturation where blue channel is greater than 100
    copy_image[:, :, 1][blue_channel > 100] -= 100
    
    # Convert the modified image to grayscale
    gray = cv2.cvtColor(copy_image, cv2.COLOR_RGB2GRAY)
    
    # Threshold the grayscale image
    _, threshold_image = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    
    # Convert the threshold image to BGR
    colored_image_land = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
    
    # Define the new color (blue)
    new_color = (0, 0, 255)
    
    # Find white pixels in the threshold image
    white_pixels = np.where(threshold_image == 255)
    
    # Apply the new color to the white pixels
    colored_image_land[white_pixels] = new_color
    
    return colored_image_land

def forest(path):
    # Read the image
    original_image = cv2.imread(path)
    
    # Make a copy of the original image
    copy_image = original_image.copy()
    
    # Convert the image from BGR to HSV
    copy_hsb = cv2.cvtColor(copy_image, cv2.COLOR_BGR2HSV)
    
    # Extract the blue channel
    blue_channel = copy_hsb[:, :, 0]
    
    # Decrease saturation where blue channel is greater than 100
    copy_image[:, :, 1][blue_channel > 100] -= 100
    
    # Invert the image colors
    copy_image = cv2.bitwise_not(copy_image)
    
    # Convert the modified image to grayscale
    gray = cv2.cvtColor(copy_image, cv2.COLOR_RGB2GRAY)
    
    # Threshold the grayscale image
    _, threshold_image = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    
    # Convert the threshold image to BGR
    colored_image_forest = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
    
    # Define the new color (green)
    new_color = (0, 255, 0)
    
    # Find white pixels in the threshold image
    white_pixels = np.where(threshold_image == 255)
    
    # Apply the new color to the white pixels
    colored_image_forest[white_pixels] = new_color
    
    return colored_image_forest

def water(path):
    # Read the image
    image = cv2.imread(path)
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the lower and upper bounds for water color in RGB
    lower_bound = np.array([70, 89, 87])
    upper_bound = np.array([200, 201, 200])
    
    # Create a mask for water regions
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    
    # Erode the mask to refine the regions
    kernel = np.ones((10, 10), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Convert the mask to a colored image
    binary_mask_gray = cv2.cvtColor(eroded_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the water color (blue) to the mask
    binary_mask_gray[np.where((binary_mask_gray == [255, 255, 255]).all(axis=2))] = [255, 0, 0]
    
    return binary_mask_gray

def forest_segmentation(input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (with case-insensitive extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            path = os.path.join(input_folder, filename)
            
            # Apply the segmentation functions
            water_seg = water(path)
            land_seg = land(path)
            forest_seg = forest(path)
            soil_seg = soil(path)
            
            # Merge the results
            water_land = cv2.addWeighted(water_seg, 1, land_seg, 1, 0)
            soil_forest = cv2.addWeighted(soil_seg, 1, forest_seg, 1, 0)
            final_image = cv2.addWeighted(water_land, 1, soil_forest, 1, 0)
            
            # Save the final merged image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, final_image)

# Examples of Folder Path
input_folder = "destination\\"  # Input Folder
output_folder = "destination\\"  # Output Folder

forest_segmentation(input_folder, output_folder)
