import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# Function to load Landsat 7 image from an uploaded file
def load_image(file):
    try:
        with MemoryFile(file) as memfile:
            with memfile.open() as src:
                image = src.read(
                    out_shape=(src.count, src.height, src.width),
                    resampling=Resampling.bilinear
                )
                image = image.transpose(1, 2, 0)  # Convert to HWC format
                return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to correct missing scan lines using nearest neighbor interpolation
def correct_missing_scanlines(image):
    corrected_image = np.copy(image)
    
    # Iterate over each band in the image
    for band in range(image.shape[2]):
        band_data = image[:, :, band]
        zero_indices = np.argwhere(band_data == 0)
        
        # Iterate over each zero pixel
        for row, col in zero_indices:
            # Get the nearest non-zero neighbors
            neighbors = []
            
            # Check above
            if row > 0 and band_data[row - 1, col] != 0:
                neighbors.append(band_data[row - 1, col])
                
            # Check below
            if row < band_data.shape[0] - 1 and band_data[row + 1, col] != 0:
                neighbors.append(band_data[row + 1, col])
                
            # Check left
            if col > 0 and band_data[row, col - 1] != 0:
                neighbors.append(band_data[row, col - 1])
                
            # Check right
            if col < band_data.shape[1] - 1 and band_data[row, col + 1] != 0:
                neighbors.append(band_data[row, col + 1])
            
            # If neighbors are found, replace the zero pixel with the average of neighbors
            if neighbors:
                corrected_image[row, col, band] = np.mean(neighbors)
    
    return corrected_image

# Streamlit app layout
st.title("Landsat 7 Image Correction Using Nearest Neighbor Interpolation")

uploaded_file = st.file_uploader("Upload a Landsat 7 Image", type=["tif", "tiff"])

if uploaded_file:
    image = load_image(uploaded_file)
    if image is not None:
        st.image(image, caption="Original Image with Missing Scan Lines", use_column_width=True)
        
        if st.button("Correct Image"):
            corrected_image = correct_missing_scanlines(image)
            st.image(corrected_image, caption="Corrected Image", use_column_width=True)
            
            # Provide download button for corrected image
            corrected_image_bytes = corrected_image.tobytes()
            st.download_button("Download Corrected Image", data=corrected_image_bytes, file_name="corrected_image.tif")
    else:
        st.error("Failed to load the image. Please ensure the file is a valid GeoTIFF.")
