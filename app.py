import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧿",
    layout="centered"
)

st.title("🧿 Tamga3D")
st.write("2D-изображение ковра → 3D-рельеф")

uploaded_file = st.file_uploader(
    "Загрузить изображение ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("1. Выбери орнамент")

    st.write(
        "Коснись изображения по очереди в четырёх углах "
        "области, которую хочешь превратить в 3D."
    )

    if "points" not in st.session_state:
        st.session_state.points = []

    coordinates = streamlit_image_coordinates(
        image,
        key="carpet_image"
    )

    if coordinates:
        point = (coordinates["x"], coordinates["y"])

        if not st.session_state.points or point != st.session_state.points[-1]:
            st.session_state.points.append(point)

    st.write(
        f"Выбрано точек: {len(st.session_state.points)} / 4"
    )

    if len(st.session_state.points) == 4:

        xs = [p[0] for p in st.session_state.points]
        ys = [p[1] for p in st.session_state.points]

        left = max(0, min(xs))
        right = min(image.width, max(xs))
        top = max(0, min(ys))
        bottom = min(image.height, max(ys))

        if right > left and bottom > top:

            selected = image.crop(
                (left, top, right, bottom)
            )

            st.subheader("2. Выбранный орнамент")

            st.image(selected)

            # Уменьшаем изображение для быстрой обработки
            max_size = 120

            selected.thumbnail(
                (max_size, max_size)
            )

            # Переводим изображение в оттенки серого.
            # Светлые участки будут выше,
            # тёмные — ниже.
            gray = ImageOps.grayscale(selected)

            gray = ImageEnhance.Contrast(gray).enhance(1.5)

            height_map = np.array(gray).astype(float)

            # Нормализация высоты
            height_map = (
                height_map - height_map.min()
            )

            if height_map.max() > 0:
                height_map = (
                    height_map / height_map.max()
                )

            # Усиливаем глубину рельефа
            height_map = height_map ** 1.5

            rows, cols = height_map.shape

            x = np.linspace(
                0,
                1,
                cols
            )

            y = np.linspace(
                0,
                1,
                rows
            )

            X, Y = np.meshgrid(x, y)

            Z = height_map * 0.35

            st.subheader("3. 3D-рельеф")

            fig = go.Figure(
                data=[
                    go.Surface(
                        x=X,
                        y=Y,
                        z=Z
                    )
                ]
            )

            fig.update_layout(
                height=500,
                margin=dict(
                    l=0,
                    r=0,
                    t=0,
                    b=0
                ),
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Высота"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                "Это первая демонстрационная версия "
                "3D-рельефа. Высота пока рассчитывается "
                "по яркости изображения."
            )

            # Сохранение выбранного орнамента
            buffer = BytesIO()

            selected.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                label="💾 Сохранить выбранный орнамент",
                data=buffer.getvalue(),
                file_name="tamga3d_ornament.png",
                mime="image/png"
            )

    if st.button("🔄 Начать выбор заново"):
        st.session_state.points = []
        st.rerun()
            
