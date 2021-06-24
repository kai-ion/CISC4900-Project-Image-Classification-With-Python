import cv2
import streamlit as st
import numpy as np 
from PIL import Image

#function to convert the image into a drawing, with four filters to choose from
def ImageToDrawing (img, Drawing):

    #convert the image into a grayscale image via cvtColor from opencv
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #using GaussianBlur to apply gray_blur to the image
    if Drawing == "Sketch":
        
        value = st.sidebar.slider('Tune the brightness of your sketch (the higher the value, the brighter your sketch)', 0.0, 300.0, 250.0)
        kernel = st.sidebar.slider('Tune the boldness of the edges of your sketch (the higher the value, the bolder the edges)', 1, 99, 25, step=2)

        # Blur the image using Gaussian Blur 
        gray_blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        Drawing = cv2.divide(gray, gray_blur, scale=value)

    if Drawing == "Edge Detection and Enhancement":
        
       
        smooth = st.sidebar.slider('Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)
        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3, step =2)
        edge_preserve = st.sidebar.slider('Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)', 0.0, 1.0, 0.5)
        
        gray = cv2.medianBlur(gray, kernel) 
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
 										cv2.THRESH_BINARY, 9, 9) 
    
        color = cv2.detailEnhance(img, sigma_s=smooth, sigma_r=edge_preserve)
        Drawing = cv2.bitwise_and(color, color, mask=edges) 

    if Drawing == "Pencil Edges":
    
        kernel = st.sidebar.slider('Tune the sharpness of the sketch (the lower the value, the sharper it is)', 1, 99, 25, step=2)
        laplacian_filter = st.sidebar.slider('Tune the edge detection power (the higher the value, the more powerful it is)', 3, 9, 3, step =2)
        noise_reduction = st.sidebar.slider('Tune the noise effects of your sketch (the higher the value, the noisier it is)', 10, 255, 150)
        
        gray = cv2.medianBlur(gray, kernel) 
        edges = cv2.Laplacian(gray, -1, ksize=laplacian_filter)

        
        edges_inv = 255-edges
    
        dummy, Drawing = cv2.threshold(edges_inv, noise_reduction, 255, cv2.THRESH_BINARY)
        

    if Drawing == "Bilateral Filter":
        
        
       
        smooth = st.sidebar.slider('Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 5, step=2)
        kernel = st.sidebar.slider('Tune the sharpness of the image (the lower the value, the sharper it is)', 1, 21, 3, step =2)
        edge_preserve = st.sidebar.slider('Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)', 1, 100, 50)
       
        gray = cv2.medianBlur(gray, kernel) 
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
 										cv2.THRESH_BINARY, 9, 9)
    
        color = cv2.bilateralFilter(img, smooth, edge_preserve, smooth) 
        Drawing = cv2.bitwise_and(color, color, mask=edges) 
    
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return Drawing

###############################################################################
    
st.write("""
          # Turn Your Image Into a Drawing

          """
          )

st.write("This is an app to turn your photos into Drawings")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    
    option = st.sidebar.selectbox(
    'Which filters would you like to apply?',
    ('Sketch', 'Edge Detection and Enhancement', 'Pencil Edges', 'Bilateral Filter'))
    
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    st.text("Your New image")
    Drawing = ImageToDrawing(img, option)
    
    st.image(Drawing, use_column_width=True)