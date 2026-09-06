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
st.write("Превращаем 2D-фрагмент ковра в тонкую цветную 3D-модель")

uploaded_file = st.file_uploader(
    "Загрузить изображение ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("1. Выбери орнамент")

    st.write(
        "Коснись изображения по очереди в четырёх углах "
        "нужной области."
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

            # ==========================================
            # ОРИГИНАЛЬНЫЙ ФРАГМЕНТ
            # ==========================================

            selected_original = image.crop(
                (left, top, right, bottom)
            )

            st.subheader("2. Выбранный фрагмент")

            st.image(
                selected_original,
                caption="Оригинальные цвета ковра"
            )

            # ==========================================
            # ПОДГОТОВКА МОДЕЛИ
            # ==========================================

            texture = selected_original.copy()

            # Размер сетки
            max_size = 120

            texture.thumbnail(
                (max_size, max_size)
            )

            texture_array = np.array(
                texture
            )

            rows, cols, _ = texture_array.shape

            # ==========================================
            # КАРТА ВЫСОТЫ
            # ==========================================

            gray = ImageOps.grayscale(
                texture
            )

            gray = ImageEnhance.Contrast(
                gray
            ).enhance(1.3)

            height_map = np.array(
                gray,
                dtype=float
            )

            height_map -= height_map.min()

            if height_map.max() > 0:
                height_map /= height_map.max()

            # ------------------------------------------
            # ОЧЕНЬ МАЛЕНЬКАЯ ТОЛЩИНА
            # ------------------------------------------

            # Это реальная толщина модели.
            # Чем меньше число, тем тоньше ковёр.

            carpet_thickness = 0.025

            # Небольшой рельеф орнамента
            relief_height = 0.015

            Z_bottom = np.zeros_like(
                height_map
            )

            Z_top = (
                carpet_thickness
                + height_map * relief_height
            )

            # ==========================================
            # КООРДИНАТЫ
            # ==========================================

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

            X, Y = np.meshgrid(
                x,
                y
            )

            # ==========================================
            # ЦВЕТА ОРИГИНАЛЬНОГО КОВРА
            # ==========================================

            rgb = texture_array.reshape(
                -1,
                3
            )

            vertex_colors = [
                f"rgb({int(r)},{int(g)},{int(b)})"
                for r, g, b in rgb
            ]

            # ==========================================
            # ТРЕУГОЛЬНИКИ
            # ==========================================

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

            # ==========================================
            # 3D-ПРОСМОТР
            # ==========================================

            st.subheader(
                "3. Тонкая цветная 3D-модель"
            )

            fig = go.Figure()

            # ------------------------------------------
            # ВЕРХНЯЯ ПОВЕРХНОСТЬ
            # ------------------------------------------

            fig.add_trace(
                go.Mesh3d(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z_top.flatten(),

                    i=top_i,
                    j=top_j,
                    k=top_k,

                    # НАСТОЯЩИЕ ЦВЕТА ФОТО
                    vertexcolor=vertex_colors,

                    flatshading=False,

                    lighting=dict(
                        ambient=0.7,
                        diffuse=0.8,
                        specular=0.15,
                        roughness=0.9
                    ),

                    lightposition=dict(
                        x=2,
                        y=2,
                        z=3
                    ),

                    hoverinfo="skip"
                )
            )

            # ==========================================
            # НИЖНЯЯ ПОВЕРХНОСТЬ
            # ==========================================

            bottom_offset = rows * cols

            all_x = np.concatenate([
                X.flatten(),
                X.flatten()
            ])

            all_y = np.concatenate([
                Y.flatten(),
                Y.flatten()
            ])

            all_z = np.concatenate([
                Z_top.flatten(),
                Z_bottom.flatten()
            ])

            # ==========================================
            # БОКОВЫЕ СТЕНКИ
            # ==========================================

            side_i = []
            side_j = []
            side_k = []

            # Передняя и задняя стороны
            for c in range(cols - 1):

                # Передняя
                a = c
                b = c + 1

                side_i.extend([
                    a,
                    a
                ])

                side_j.extend([
                    b,
                    bottom_offset + b
                ])

                side_k.extend([
                    bottom_offset + b,
                    bottom_offset + a
                ])

                # Задняя
                a = (
                    (rows - 1) * cols
                    + c
                )

                b = a + 1

                side_i.extend([
                    a,
                    a
                ])

                side_j.extend([
                    bottom_offset + a,
                    b
                ])

                side_k.extend([
                    b,
                    bottom_offset + b
                ])

            # Левая и правая стороны
            for r in range(rows - 1):

                # Левая
                a = r * cols
                b = (r + 1) * cols

                side_i.extend([
                    a,
                    a
                ])

                side_j.extend([
                    bottom_offset + a,
                    b
                ])

                side_k.extend([
                    b,
                    bottom_offset + b
                ])

                # Правая
                a = (
                    r * cols
                    + cols - 1
                )

                b = (
                    (r + 1) * cols
                    + cols - 1
                )

                side_i.extend([
                    a,
                    a
                ])

                side_j.extend([
                    b,
                    bottom_offset + b
                ])

                side_k.extend([
                    bottom_offset + b,
                    bottom_offset + a
                ])

            # ------------------------------------------
            # БОКОВАЯ ПОВЕРХНОСТЬ
            # ------------------------------------------

            fig.add_trace(
                go.Mesh3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,

                    i=side_i,
                    j=side_j,
                    k=side_k,

                    color="rgb(90,60,40)",

                    flatshading=True,

                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.7,
                        specular=0.1
                    ),

                    hoverinfo="skip"
                )
            )

            # ==========================================
            # ВАЖНО: МОДЕЛЬ ВЫГЛЯДИТ ТОНКОЙ
            # ==========================================

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

                    # Соотношение сторон:
                    # ширина : длина : толщина
                    aspectmode="manual",

                    aspectratio=dict(
                        x=1,
                        y=1,
                        z=0.08
                    ),

                    camera=dict(
                        eye=dict(
                            x=1.5,
                            y=1.5,
                            z=1.1
                        )
                    )
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                "Готово! Получена тонкая цветная "
                "3D-модель выбранного фрагмента."
            )

            # ==========================================
            # СОЗДАНИЕ OBJ
            # ==========================================

            st.subheader(
                "4. Скачать модель"
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

            # Верх
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

            # Низ
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

            # Левая сторона
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

            # Правая сторона
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

            # ==========================================
            # OBJ + UV
            # ==========================================

            obj_lines = []

            obj_lines.append(
                "# Tamga3D carpet model"
            )

            obj_lines.append(
                "mtllib tamga3d_ornament.mtl"
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

            # ==========================================
            # MTL
            # ==========================================

            mtl_text = """newmtl CarpetTexture
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
illum 1
map_Kd carpet_texture.png
"""

            # ==========================================
            # PNG-ТЕКСТУРА
            # ==========================================

            texture_buffer = BytesIO()

            texture.save(
                texture_buffer,
                format="PNG"
            )

            texture_data = (
                texture_buffer.getvalue()
            )

            # ==========================================
            # ZIP
            # ==========================================

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
                label="⬇️ Скачать цветную 3D-модель",
                data=zip_buffer.getvalue(),
                file_name="tamga3d_3d_model.zip",
                mime="application/zip"
            )

            st.info(
                "ZIP содержит 3D-геометрию OBJ, материал MTL "
                "и текстуру исходного ковра."
            )

    if st.button("🔄 Начать выбор заново"):

        st.session_state.points = []

        st.rerun()
