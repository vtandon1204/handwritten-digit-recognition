import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = joblib.load('model/model.pkl')

# Page Configuration
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered", page_icon="✏️")

# CSS Styling for Modern Look
st.markdown(
    """
    <style>
    .main-container {
        background-color: #F7F7F9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .header-title {
        text-align: center;
        color: #4B0082;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4B0082;
        color: white;
        text-align: center;
        padding: 10px;
    }
    .footer a {
        color: white;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
        color: lavender;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown("<h1 class='header-title'>🎨 Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("This application lets you draw a digit, processes it, and predicts it using a pre-trained MNIST model.")

# Main Container
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Draw Canvas
    st.markdown("### 🖌️ Draw Your Digit Below:")
    canvas_result = st_canvas(
        fill_color="white",  # Background color
        stroke_width=10,     # Pen stroke width
        stroke_color="black", # Pen color
        background_color="white", # Canvas background
        height=280,           # Canvas height
        width=280,            # Canvas width
        drawing_mode="freedraw", # Drawing mode
        key="canvas",
    )

    # Prediction Section
    if canvas_result.image_data is not None:
        # Convert canvas to grayscale and preprocess
        image_data = canvas_result.image_data[:, :, :3]  # Remove alpha channel
        img = Image.fromarray((255 - image_data).astype("uint8"))  # Invert colors

        # Columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Processed Image:")
            st.image(img.resize((140, 140)), caption="28x28 Processed Image", width=140)

        with col2:
            st.markdown("### Prediction:")
            if st.button("🔍 Predict"):
                try:
                    # Preprocessing function
                    def preprocess_image(img):
                        img = img.resize((28, 28))  # Resize to 28x28
                        img = img.convert('L')  # Convert to grayscale
                        img = np.array(img) / 255.0  # Normalize
                        img = img.flatten().reshape(1, -1)  # Flatten
                        return img

                    processed_img = preprocess_image(img)
                    prediction = model.predict(processed_img)
                    st.success(f"The predicted digit is: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # End Main Container
    st.markdown("</div>", unsafe_allow_html=True)

# Footer Section
st.markdown(
    """
    <div class="footer">
        <p>Developed with ❤️ by <a href="https://github.com/vtandon1204" target="_blank">Vaibhav Tandon</a>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
