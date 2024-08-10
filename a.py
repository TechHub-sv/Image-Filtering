import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('C:/Users/Viraj/OneDrive/Desktop/photo.jpg')

def display_image(img, title='Image'):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(img, kernel)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image

def apply_invert(img):
    return cv2.bitwise_not(img)

def apply_histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def apply_pencil_sketch(img):
    gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_edge_detection(img):
    return cv2.Canny(img, 100, 200)

def apply_blur(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_pixelation(img, pixel_size=10):
    height, width = img.shape[:2]
    temp = cv2.resize(img, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_image

def adjust_contrast(img, alpha=1.3):
    adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return adjusted_image

def apply_cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
    
# Apply Histogram Equalization filter
histogram_image = apply_histogram_equalization(image)
display_image(histogram_image, 'Histogram Equalization Image')
    
# Apply Pencil Sketch filter
sketch_image = apply_pencil_sketch(image)
display_image(sketch_image, 'Pencil Sketch Image')

# Apply sepia Image filter
sepia_image = apply_sepia(image)
display_image(sepia_image, 'Sepia Image')
    
# Apply Grayscale filter
gray_image = apply_grayscale(image)
display_image(gray_image, 'Grayscale Image')
    
# Apply Edge Detection filter
edges_image = apply_edge_detection(image)
display_image(edges_image, 'Edge Detection Image')
    
# Apply Gaussian Blur filter
blur_image = apply_blur(image)
display_image(blur_image, 'Gaussian Blur Image')


# Apply Invert filter
invert_image = apply_invert(image)
display_image(invert_image, 'Invert Image')
    
# Sharpening
kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)
display_image(sharpened, 'Sharpening')

# Custom Filter (example: emboss effect)
kernel_emboss = np.array([[2, 0, 0], 
                          [0, -1, 0], 
                          [0, 0, -1]])
embossed = cv2.filter2D(image, -1, kernel_emboss)
display_image(embossed, 'Emboss Filter')

# Apply Pixelation filter
pixel_image = apply_pixelation(image, pixel_size=20)
display_image(pixel_image, 'Pixelation Image')

# Adjust Brightness
bright_image = adjust_brightness(image, value=50)
display_image(bright_image, 'Bright Image')

# Adjust Contrast
contrast_image = adjust_contrast(image, alpha=1.5)
display_image(contrast_image, 'Contrast Image')
    
# Apply Cartoon Effect
cartoon_image = apply_cartoon_effect(image)
display_image(cartoon_image, 'Cartoon Image')

# Save the processed image
cv2.imwrite('srushti.jpg',sepia_image)
