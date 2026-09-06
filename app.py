import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🔷",
    layout="wide"
)

st.title("🔷 Tamga3D")
st.write("Prototype for extracting ornament from carpet images")

uploaded_file = st.file_uploader(
    "📤 Upload a carpet image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Original image")
    st.image(image, use_container_width=True)

    if st.button("⚙️ Detect ornament"):

        # Convert RGB to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Reduce image noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a black image for the detected ornament
        result = np.zeros_like(gray)

        # Keep sufficiently large contours
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 100:
                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    255,
                    2
                )

        st.subheader("Detected ornament")

        st.image(
            result,
            caption="Detected ornament / contours",
            use_container_width=True
        )

        # Prepare PNG for download
        result_image = Image.fromarray(result)

        buffer = BytesIO()
        result_image.save(buffer, format="PNG")

        st.download_button(
            label="💾 Save detected ornament",
            data=buffer.getvalue(),
            file_name="detected_ornament.png",
            mime="image/png"
  )
