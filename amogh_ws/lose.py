
import cv2
import time

# Path to your image file
image_path  = '/home/t_b/stonefish_ros/win.png'

print("ommmmmmmmmm")
# Read the image
image = cv2.imread(image_path)

# Create a named window
cv2.namedWindow('YOU LOSE', cv2.WINDOW_NORMAL)

# Move the window to position (x, y)
cv2.moveWindow('YOU LOSE', 300, 200)  # Replace 300, 200 with your desired coordinates

# Show the image
cv2.imshow('YOU LOSE', image)
import cv2

# Path to your image file
image_path  = 'image_lose.png'

print("ommmmmmmmmm")
# Read the image
image = cv2.imread(image_path)

# Create a named window
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)

# Move the window to position (x, y)
cv2.moveWindow('My Image', 300, 200)  # Replace 300, 200 with your desired coordinates

cv2.imshow('My Image', image)
cv2.waitKey(2000)  # allows OpenCV to render the window
time.sleep(2    )
cv2.destroyAllWindows()




