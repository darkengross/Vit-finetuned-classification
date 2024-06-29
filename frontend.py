import streamlit as st
import requests
from PIL import Image

# Set the title of the app
st.title("Image Classification using Vision Transformer")

# Create three columns with an empty column in the middle for spacing
col1, spacer, col2 = st.columns([2, 0.5, 1])

# Left column for image upload and classification
with col1:
    st.header("Upload and Classify Image")
    st.write("Upload an image to get the classification result.")

    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Classify"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Convert image to bytes
                uploaded_file.seek(0)
                files = {"files": (uploaded_file.name, uploaded_file, "image/jpeg")}
                
                # Make request to FastAPI backend
                response = requests.post("http://127.0.0.1:8000/uploadfiles/", files=files)
                
                if response.status_code == 200:
                    result = response.json()[0]
                    if 'error' in result:
                        st.write(f"**Filename:** {result['filename']} - Error: {result['error']}")
                    else:
                        st.write(f"**Filename:** {result['filename']}")
                        st.write(f"**Class Type:** {result['class_type']}")
                else:
                    st.write("Error in classification. Please try again.")
        else:
            st.write("Please upload at least one image.")

# Right column for displaying text
with col2:
    st.header("About the Project")
    st.write("""
    This is a web application for the finetuned vit_b_16 pytorch vision transformer on [Intel classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
    
    The Model accept any image and try classify it in 6 classes
    - forest
    - glacier
    - mountain
    - sea
    - street
    
    The model achives an accuracy of 96% on the training dataset and 94% on a test dataset
    """)

# Add some CSS to change the layout and title style
st.markdown("""
    <style>
        .stFileUploader>div>div {
            border: 2px dashed lightgreen;
        }
    </style>
    """, unsafe_allow_html=True)
