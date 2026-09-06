import streamlit as st
from PIL import Image
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧿",
    layout="centered"
)

st.title("🧿 Tamga3D")
st.write("Выбери орнамент на изображении ковра")

uploaded_file = st.file_uploader(
    "Загрузить изображение ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("1. Выбери область орнамента")

    st.write(
        "Коснись изображения по очереди в четырёх углах "
        "области, которую хочешь выделить."
    )

    if "points" not in st.session_state:
        st.session_state.points = []

    displayed_image = image.copy()

    coordinates = streamlit_image_coordinates(
        displayed_image,
        key="carpet_image"
    )

    if coordinates:
        point = (coordinates["x"], coordinates["y"])

        if not st.session_state.points or point != st.session_state.points[-1]:
            st.session_state.points.append(point)

    st.write(f"Выбрано точек: {len(st.session_state.points)} / 4")

    if st.session_state.points:
        st.write("Точки:")
        for i, point in enumerate(st.session_state.points, 1):
            st.write(f"{i}: {point}")

    if len(st.session_state.points) == 4:
        xs = [p[0] for p in st.session_state.points]
        ys = [p[1] for p in st.session_state.points]

        left = max(0, min(xs))
        right = min(image.width, max(xs))
        top = max(0, min(ys))
        bottom = min(image.height, max(ys))

        if right > left and bottom > top:
            selected = image.crop((left, top, right, bottom))

            st.subheader("2. Выбранный орнамент")

            st.image(selected, caption="Выбранная область")

            selected_array = np.array(selected)

            result = Image.fromarray(selected_array)

            from io import BytesIO

            buffer = BytesIO()
            result.save(buffer, format="PNG")

            st.download_button(
                label="💾 Сохранить орнамент",
                data=buffer.getvalue(),
                file_name="tamga3d_ornament.png",
                mime="image/png"
            )

    if st.button("🔄 Начать выбор заново"):
        st.session_state.points = []
        st.rerun()
