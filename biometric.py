import cv2
import numpy as np

# Load the three sample fingerprint images
sample_images = []
for i in range(1, 4):
    img = cv2.imread(f'{i}.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Sample image {i}.jpg not loaded. Make sure the file exists and is in the correct path.")
        exit()  # Exit the program if an image is not loaded
    sample_images.append(img)

# Load the query fingerprint image
q_i = input("Enter your image name with extension: ")
query_image = cv2.imread(q_i, cv2.IMREAD_GRAYSCALE)
if query_image is None:
    print("Query fingerprint image not loaded. Make sure the file exists and is in the correct path.")
    exit()  # Exit the program if the query image is not loaded

# Create ORB (Oriented FAST and Rotated BRIEF) detector
orb = cv2.ORB_create()

# Find keypoints and descriptors for the sample images and the query image
sample_keypoints = []
sample_descriptors = []
for img in sample_images:
    kp, des = orb.detectAndCompute(img, None)
    sample_keypoints.append(kp)
    sample_descriptors.append(des)

query_keypoints, query_descriptors = orb.detectAndCompute(query_image, None)

# Create a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors of the query image with each sample image
matches = []
for des in sample_descriptors:
    match = bf.match(des, query_descriptors)
    matches.append(match)

# Calculate the matching score for each sample image
matching_scores = [len(match) for match in matches]

# Determine the index of the best matching sample image
best_match_index = np.argmax(matching_scores)

# Define a threshold for a "good" match (you can adjust this threshold)
threshold = 250

# Check if the best match exceeds the threshold
if matching_scores[best_match_index] > threshold:
    print(f"Matched with sample fingerprint {best_match_index + 1}")
else:
    print("No match found")

# Add text to the input query image
input_image_with_text = query_image.copy()
input_image_name = q_i.split('.')[0]  # Get the image name without extension
cv2.putText(input_image_with_text, input_image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Add text to the matching result image
result_image = cv2.drawMatches(sample_images[best_match_index], sample_keypoints[best_match_index],
                              query_image, query_keypoints, matches[best_match_index], None)
result_image_with_text = result_image.copy()
result_image_name = f'result_{best_match_index + 1}.jpg'
cv2.putText(result_image_with_text, result_image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the result image with text
cv2.imwrite(result_image_name, result_image_with_text)
print(f"Result image saved as '{result_image_name}'")

# Display the input image and result image
cv2.waitKey(0)
cv2.destroyAllWindows()
