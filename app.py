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
st.write("2D-изображение ковра → 3D-рельеф → OBJ")

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

        if (
            not st.session_state.points
            or point != st.session_state.points[-1]
        ):
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

            selected_original = image.crop(
                (left, top, right, bottom)
            )

            st.subheader("2. Выбранный орнамент")

            st.image(
                selected_original,
                caption="Выбранная область"
            )

            # Уменьшаем изображение для быстрой генерации сетки
            selected = selected_original.copy()

            max_size = 120

            selected.thumbnail(
                (max_size, max_size)
            )

            # Создаём карту высоты
            gray = ImageOps.grayscale(selected)

            gray = ImageEnhance.Contrast(gray).enhance(1.5)

            height_map = np.array(
                gray,
                dtype=float
            )

            height_map -= height_map.min()

            if height_map.max() > 0:
                height_map /= height_map.max()

            # Усиливаем рельеф
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

            depth = 0.35

            Z = height_map * depth

            # -------------------------
            # 3D-ПРОСМОТР
            # -------------------------

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

            # -------------------------
            # СОЗДАНИЕ OBJ
            # -------------------------

            st.subheader("4. 3D-модель")

            vertices = []
            faces = []

            # Создаём вершины
            for r in range(rows):
                for c in range(cols):

                    vx = X[r, c]
                    vy = Y[r, c]
                    vz = Z[r, c]

                    vertices.append(
                        (vx, vy, vz)
                    )

            # Создаём треугольники
            for r in range(rows - 1):
                for c in range(cols - 1):

                    i1 = r * cols + c
                    i2 = r * cols + c + 1
                    i3 = (r + 1) * cols + c
                    i4 = (r + 1) * cols + c + 1

                    faces.append(
                        (i1 + 1, i2 + 1, i4 + 1)
                    )

                    faces.append(
                        (i1 + 1, i4 + 1, i3 + 1)
                    )

            # Записываем OBJ
            obj = []

            obj.append(
                "# Tamga3D generated OBJ"
            )

            obj.append(
                "# 2D carpet ornament converted to 3D"
            )

            for vertex in vertices:

                obj.append(
                    f"v {vertex[0]:.6f} "
                    f"{vertex[1]:.6f} "
                    f"{vertex[2]:.6f}"
                )

            for face in faces:

                obj.append(
                    f"f {face[0]} "
                    f"{face[1]} "
                    f"{face[2]}"
                )

            obj_text = "\n".join(obj)

            st.success(
                "3D-сетка создана!"
            )

            st.download_button(
                label="⬇️ Скачать 3D-модель (.OBJ)",
                data=obj_text,
                file_name="tamga3d_ornament.obj",
                mime="text/plain"
            )

            # -------------------------
            # СОХРАНЕНИЕ ОРНАМЕНТА
            # -------------------------

            buffer = BytesIO()

            selected_original.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                label="💾 Скачать орнамент (.PNG)",
                data=buffer.getvalue(),
                file_name="tamga3d_ornament.png",
                mime="image/png"
            )

            st.info(
                "Это первая версия генерации 3D-сетки. "
                "Сейчас высота рельефа рассчитывается "
                "по яркости изображения."
            )

    if st.button("🔄 Начать выбор заново"):

        st.session_state.points = []

        st.rerun()
