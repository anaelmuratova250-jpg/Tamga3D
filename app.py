import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go
from io import BytesIO
import zipfile

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧿",
    layout="centered"
)

st.title("🧿 Tamga3D")
st.write("2D-изображение ковра → тонкая цветная 3D-модель")

uploaded_file = st.file_uploader(
    "Загрузить изображение ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("1. Выбери орнамент")

    st.write(
        "Коснись изображения по очереди в четырёх углах "
        "области ковра, которую хочешь превратить в 3D-модель."
    )

    if "points" not in st.session_state:
        st.session_state.points = []

    coordinates = streamlit_image_coordinates(
        image,
        key="carpet_image"
    )

    if coordinates:
        point = (
            coordinates["x"],
            coordinates["y"]
        )

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

            # Оригинальный выбранный фрагмент
            selected_original = image.crop(
                (left, top, right, bottom)
            )

            st.subheader("2. Выбранный фрагмент")

            st.image(
                selected_original,
                caption="Оригинальные цвета ковра"
            )

            # --------------------------------
            # ПОДГОТОВКА ТЕКСТУРЫ
            # --------------------------------

            texture = selected_original.copy()

            max_size = 120

            texture.thumbnail(
                (max_size, max_size)
            )

            texture_array = np.array(texture)

            rows, cols, _ = texture_array.shape

            # --------------------------------
            # КАРТА ВЫСОТЫ
            # --------------------------------

            gray = ImageOps.grayscale(texture)

            gray = ImageEnhance.Contrast(
                gray
            ).enhance(1.4)

            height_map = np.array(
                gray,
                dtype=float
            )

            height_map -= height_map.min()

            if height_map.max() > 0:
                height_map /= height_map.max()

            # Небольшая высота, как у ковра
            relief_height = 0.08

            Z_top = (
                height_map * relief_height
                + relief_height
            )

            # Нижняя сторона ковра
            Z_bottom = np.zeros_like(Z_top)

            # --------------------------------
            # КООРДИНАТЫ
            # --------------------------------

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

            st.subheader("3. Цветная 3D-модель")

            # --------------------------------
            # ЦВЕТНАЯ ВЕРХНЯЯ ПОВЕРХНОСТЬ
            # --------------------------------

            rgb_values = texture_array.reshape(
                -1,
                3
            )

            vertex_colors = [
                f"rgb({r},{g},{b})"
                for r, g, b in rgb_values
            ]

            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z_top.flatten()

            # Индексы верхней поверхности
            top_i = []
            top_j = []
            top_k = []

            for r in range(rows - 1):
                for c in range(cols - 1):

                    a = r * cols + c
                    b = a + 1
                    d = (r + 1) * cols + c
                    e = d + 1

                    top_i.append(a)
                    top_j.append(b)
                    top_k.append(e)

                    top_i.append(a)
                    top_j.append(e)
                    top_k.append(d)

            # Верхняя поверхность
            fig = go.Figure()

            fig.add_trace(
                go.Mesh3d(
                    x=x_flat,
                    y=y_flat,
                    z=z_flat,
                    i=top_i,
                    j=top_j,
                    k=top_k,
                    intensity=np.arange(
                        len(z_flat)
                    ),
                    colorscale=[
                        [
                            0,
                            "rgb(0,0,0)"
                        ],
                        [
                            1,
                            "rgb(255,255,255)"
                        ]
                    ],
                    showscale=False,
                    flatshading=False,
                    hoverinfo="skip"
                )
            )

            # --------------------------------
            # НИЖНЯЯ ПОВЕРХНОСТЬ
            # --------------------------------

            bottom_offset = len(z_flat)

            bottom_x = x_flat
            bottom_y = y_flat
            bottom_z = Z_bottom.flatten()

            all_x = np.concatenate(
                [x_flat, bottom_x]
            )

            all_y = np.concatenate(
                [y_flat, bottom_y]
            )

            all_z = np.concatenate(
                [z_flat, bottom_z]
            )

            # --------------------------------
            # БОКОВЫЕ СТЕНКИ
            # --------------------------------

            side_i = []
            side_j = []
            side_k = []

            # Передняя/задняя стороны
            for c in range(cols - 1):

                a = c
                b = c + 1

                side_i.append(a)
                side_j.append(b)
                side_k.append(
                    bottom_offset + b
                )

                side_i.append(a)
                side_j.append(
                    bottom_offset + b
                )
                side_k.append(
                    bottom_offset + a
                )

                a = (
                    (rows - 1) * cols
                    + c
                )

                b = a + 1

                side_i.append(a)
                side_j.append(
                    bottom_offset + a
                )
                side_k.append(b)

                side_i.append(b)
                side_j.append(
                    bottom_offset + a
                )
                side_k.append(
                    bottom_offset + b
                )

            # Левая/правая стороны
            for r in range(rows - 1):

                a = r * cols
                b = (r + 1) * cols

                side_i.append(a)
                side_j.append(
                    bottom_offset + a
                )
                side_k.append(b)

                side_i.append(b)
                side_j.append(
                    bottom_offset + a
                )
                side_k.append(
                    bottom_offset + b
                )

                a = r * cols + cols - 1
                b = (r + 1) * cols + cols - 1

                side_i.append(a)
                side_j.append(b)
                side_k.append(
                    bottom_offset + a
                )

                side_i.append(b)
                side_j.append(
                    bottom_offset + b
                )
                side_k.append(
                    bottom_offset + a
                )

            # --------------------------------
            # ДОБАВЛЯЕМ БОКОВЫЕ СТЕНКИ
            # --------------------------------

            fig.add_trace(
                go.Mesh3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,
                    i=side_i,
                    j=side_j,
                    k=side_k,
                    color="rgb(120,80,50)",
                    opacity=1.0,
                    flatshading=True,
                    hoverinfo="skip"
                )
            )

            fig.update_layout(
                height=550,
                margin=dict(
                    l=0,
                    r=0,
                    t=20,
                    b=0
                ),
                scene=dict(
                    xaxis_title="Ширина",
                    yaxis_title="Длина",
                    zaxis_title="Толщина",
                    aspectmode="auto"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                "Готово! Это тонкая 3D-модель "
                "выбранного фрагмента ковра."
            )

            # --------------------------------
            # СОЗДАНИЕ OBJ
            # --------------------------------

            st.subheader(
                "4. Скачать 3D-модель"
            )

            vertices = []

            # Верхние вершины
            for r in range(rows):
                for c in range(cols):

                    vertices.append(
                        (
                            X[r, c],
                            Y[r, c],
                            Z_top[r, c]
                        )
                    )

            # Нижние вершины
            for r in range(rows):
                for c in range(cols):

                    vertices.append(
                        (
                            X[r, c],
                            Y[r, c],
                            Z_bottom[r, c]
                        )
                    )

            faces = []

            # Верхняя поверхность
            for r in range(rows - 1):
                for c in range(cols - 1):

                    a = r * cols + c + 1
                    b = a + 1
                    d = (r + 1) * cols + c + 1
                    e = d + 1

                    faces.append(
                        (a, b, e)
                    )

                    faces.append(
                        (a, e, d)
                    )

            # Нижняя поверхность
            bottom_start = rows * cols

            for r in range(rows - 1):
                for c in range(cols - 1):

                    a = (
                        bottom_start
                        + r * cols
                        + c
                        + 1
                    )

                    b = a + 1

                    d = (
                        bottom_start
                        + (r + 1) * cols
                        + c
                        + 1
                    )

                    e = d + 1

                    faces.append(
                        (a, e, b)
                    )

                    faces.append(
                        (a, d, e)
                    )

            # Боковые стенки
            for c in range(cols - 1):

                top_a = c + 1
                top_b = c + 2

                bottom_a = (
                    bottom_start
                    + c
                    + 1
                )

                bottom_b = (
                    bottom_start
                    + c
                    + 2
                )

                faces.append(
                    (
                        top_a,
                        bottom_a,
                        top_b
                    )
                )

                faces.append(
                    (
                        top_b,
                        bottom_a,
                        bottom_b
                    )
                )

            for c in range(cols - 1):

                top_a = (
                    (rows - 1) * cols
                    + c
                    + 1
                )

                top_b = top_a + 1

                bottom_a = (
                    bottom_start
                    + (rows - 1) * cols
                    + c
                    + 1
                )

                bottom_b = bottom_a + 1

                faces.append(
                    (
                        top_a,
                        top_b,
                        bottom_a
                    )
                )

                faces.append(
                    (
                        top_b,
                        bottom_b,
                        bottom_a
                    )
                )

            for r in range(rows - 1):

                top_a = r * cols + 1
                top_b = (r + 1) * cols + 1

                bottom_a = (
                    bottom_start
                    + r * cols
                    + 1
                )

                bottom_b = (
                    bottom_start
                    + (r + 1) * cols
                    + 1
                )

                faces.append(
                    (
                        top_a,
                        bottom_a,
                        top_b
                    )
                )

                faces.append(
                    (
                        top_b,
                        bottom_a,
                        bottom_b
                    )
                )

            for r in range(rows - 1):

                top_a = (
                    r * cols
                    + cols
                )

                top_b = (
                    (r + 1) * cols
                    + cols
                )

                bottom_a = (
                    bottom_start
                    + r * cols
                    + cols
                )

                bottom_b = (
                    bottom_start
                    + (r + 1) * cols
                    + cols
                )

                faces.append(
                    (
                        top_a,
                        top_b,
                        bottom_a
                    )
                )

                faces.append(
                    (
                        top_b,
                        bottom_b,
                        bottom_a
                    )
                )

            # --------------------------------
            # OBJ
            # --------------------------------

            obj_lines = []

            obj_lines.append(
                "# Tamga3D textured carpet model"
            )

            for v in vertices:

                obj_lines.append(
                    f"v {v[0]:.6f} "
                    f"{v[1]:.6f} "
                    f"{v[2]:.6f}"
                )

            # UV-координаты
            for r in range(rows):

                v = 1.0 - (
                    r / max(1, rows - 1)
                )

                for c in range(cols):

                    u = c / max(
                        1,
                        cols - 1
                    )

                    obj_lines.append(
                        f"vt {u:.6f} {v:.6f}"
                    )

            obj_lines.append(
                "usemtl CarpetTexture"
            )

            # Грани с UV
            for face in faces:

                obj_lines.append(
                    "f "
                    + " ".join(
                        f"{idx}/{idx}"
                        for idx in face
                    )
                )

            obj_text = "\n".join(
                obj_lines
            )

            # --------------------------------
            # MTL
            # --------------------------------

            mtl_text = """newmtl CarpetTexture
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
illum 1
map_Kd carpet_texture.png
"""

            # --------------------------------
            # ТЕКСТУРА PNG
            # --------------------------------

            texture_buffer = BytesIO()

            texture.save(
                texture_buffer,
                format="PNG"
            )

            texture_data = (
                texture_buffer.getvalue()
            )

            # --------------------------------
            # ZIP
            # --------------------------------

            zip_buffer = BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                zip_file.writestr(
                    "tamga3d_ornament.obj",
                    obj_text
                )

                zip_file.writestr(
                    "tamga3d_ornament.mtl",
                    mtl_text
                )

                zip_file.writestr(
                    "carpet_texture.png",
                    texture_data
                )

            st.download_button(
                label="⬇️ Скачать цветную 3D-модель (.ZIP)",
                data=zip_buffer.getvalue(),
                file_name="tamga3d_3d_model.zip",
                mime="application/zip"
            )

            st.info(
                "ZIP содержит OBJ-модель, материал MTL "
                "и оригинальную текстуру ковра."
            )

    if st.button("🔄 Начать выбор заново"):

        st.session_state.points = []

        st.rerun()
