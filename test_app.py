import streamlit as st
import numpy as np
from skimage import io, color, morphology
from skimage.draw import polygon_perimeter
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt

# Function to perform convex hull operation
def convex_hull(image):
    binary = image > 0
    return morphology.convex_hull_image(binary)

# Function to perform skeletonization operation
def skeletonize(image):
    return morphology.skeletonize(image > 0)

# Function to perform active contour operation
def active_contour_segmentation(image):
    # Convert image to grayscale if it's not already
    if image.ndim == 3:
        image = color.rgb2gray(image)

    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(image, init, alpha=0.015, beta=10, gamma=0.001)

    return snake

# CSS styles with light blue background
st.markdown(
    """
    <style>
        /* Center align title */
        .css-1lq13yl {
            text-align: center;
        }

        /* Style sidebar */
        .sidebar .sidebar-content {
            background-color: #d4e7f7 !important;
            padding: 20px;
            border-radius: 10px;
        }

        /* Style uploaded image */
        .st-ef {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
        }

        /* Style subheaders */
        .st-b9 {
            color: #333333;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        /* Style multiselect */
        .multiselect {
            margin-top: 10px;
            margin-bottom: 20px;
        }

        /* Style image results */
        .st-dz {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Light blue background */
        .stApp {
            background-color: #FFC0CB !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit application
def main():
    st.title("Proses perubahan gambar")

    # Upload image
    st.sidebar.title("Masukkan Gambar")
    uploaded_file = st.sidebar.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = io.imread(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform operations
        st.sidebar.title("Pilih Perubahan")
        operations = st.sidebar.multiselect(
            "Choose operation(s):",
            ["Convex Hull", "Skeletonization", "Active Contour"]
        )

        if "Convex Hull" in operations:
            st.subheader("Convex Hull")
            result = convex_hull(image)
            st.image(result, caption="Convex Hull", use_column_width=True)

        if "Skeletonization" in operations:
            st.subheader("Skeletonization")
            gray = color.rgb2gray(image)
            result = skeletonize(gray)
            st.image(result, caption="Skeletonization", use_column_width=True)

        if "Active Contour" in operations:
            st.subheader("Active Contour")
            result = active_contour_segmentation(image)
            
            fig, ax = plt.subplots()
            ax.imshow(image, cmap=plt.cm.gray)
            snake = np.array(result)
            ax.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            st.pyplot(fig)

# Run the application
if __name__ == "__main__":
    main()
