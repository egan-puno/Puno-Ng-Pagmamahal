import cv2
import numpy as np
import os

def generate_annotations(input_folder, output_folder, image_size):
    # Define class indices and their corresponding colors
    class_indices = {
        0: (0, 0, 255),       # Water - Blue
        1: (0, 255, 0),       # Forest - Green
        2: (255, 0, 0),       # Land - Red
        3: (255, 0, 255),      # Shallow_Water - Magenta
        4: (255, 255, 0)    # Soil_Erosion - Yellow
    }

    # Iterate over each image in the input folder
    for image_name in os.listdir(input_folder):
        if image_name.endswith(('.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg')):
            # Load segmented image
            image_path = os.path.join(input_folder, image_name)
            image_name_no_ext = os.path.splitext(image_name)[0]
            segmented_image = cv2.imread(image_path)

            # Convert the image from BGR to RGB
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

            annotations = {}

            # Iterate over each class
            for class_id, color in class_indices.items():
                # Create a mask for the current color
                mask = cv2.inRange(segmented_image, np.array(color), np.array(color))

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) < 50: # Skip contours with less than 50 points
                        continue  
                    contour = contour.squeeze()  # Flatten contour
                    normalized_contour = contour.astype(np.float32)
                    normalized_contour[:, 0] /= image_size[0]  # Normalize x coordinates
                    normalized_contour[:, 1] /= image_size[1]  # Normalize y coordinates
                    annotation = f"{class_id} {' '.join(map(str, normalized_contour.flatten()))}"
                    annotations.setdefault(class_id, []).append(annotation)

            # Write annotations to a text file in the output folder
            output_path = os.path.join(output_folder, f"{image_name_no_ext}.txt")
            with open(output_path, "w") as f:
                for class_id, class_annotations in annotations.items():
                    f.write("\n".join(class_annotations) + "\n")

            print(f"Annotations saved to {output_path}")

# Example usage:
img_width = 5280
img_height = 3956
input_folder = "dir/input"
output_folder = "dir/output/"
image_size = (5280, 3956)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

generate_annotations(input_folder, output_folder, image_size)
