import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import zipfile


# ============================================================
# TAMGA3D
# ============================================================

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧿",
    layout="wide"
)

st.title("🧿 Tamga3D")
st.write(
    "Фото → выбор орнамента → цвет → тонкая 3D-модель → OBJ для 3ds Max"
)


# ============================================================
# SESSION STATE
# ============================================================

if "points" not in st.session_state:
    st.session_state.points = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# ============================================================
# ЗАГРУЗКА
# ============================================================

uploaded = st.file_uploader(
    "Загрузите фотографию ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Загрузите фотографию ковра.")
    st.stop()


# ============================================================
# ОТКРЫВАЕМ ИЗОБРАЖЕНИЕ
# ============================================================

data = np.asarray(
    bytearray(uploaded.read()),
    dtype=np.uint8
)

image = cv2.imdecode(
    data,
    cv2.IMREAD_COLOR
)

if image is None:
    st.error("Не удалось открыть изображение.")
    st.stop()

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# УМЕНЬШЕНИЕ ФОТО
# ============================================================

max_size = 900

height, width = image.shape[:2]

scale = min(
    1.0,
    max_size / max(height, width)
)

if scale < 1.0:

    image = cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale)
        ),
        interpolation=cv2.INTER_AREA
    )


height, width = image.shape[:2]


# ============================================================
# НАСТРОЙКИ
# ============================================================

st.sidebar.header("⚙️ Настройки")

color_count = st.sidebar.slider(
    "Количество цветов",
    2,
    8,
    5
)

min_area = st.sidebar.slider(
    "Минимальный размер элемента",
    20,
    1000,
    100,
    10
)

carpet_thickness_mm = st.sidebar.slider(
    "Толщина ковра",
    1.0,
    8.0,
    3.0,
    0.5
)

ornament_height_mm = st.sidebar.slider(
    "Высота орнамента",
    0.5,
    6.0,
    2.0,
    0.5
)

ignore_background = st.sidebar.checkbox(
    "Не делать самый большой цвет объёмным",
    value=True
)


# ============================================================
# ВЫБОР ОБЛАСТИ
# ============================================================

st.header("1. Выберите часть ковра")

st.write(
    "Нажмите 4 точки вокруг нужного орнамента."
)

try:

    from streamlit_image_coordinates import (
        streamlit_image_coordinates
    )

    clicked = streamlit_image_coordinates(
        image,
        key="tamga_selector"
    )

except Exception as error:

    st.error(
        "Не удалось загрузить инструмент выбора изображения."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# СОХРАНЯЕМ КЛИК
# ============================================================

if clicked is not None:

    point = (
        int(clicked["x"]),
        int(clicked["y"])
    )

    if point != st.session_state.last_click:

        if len(st.session_state.points) < 4:

            st.session_state.points.append(point)

        st.session_state.last_click = point

        st.rerun()


# ============================================================
# ПОКАЗ ТОЧЕК
# ============================================================

st.write(
    "Выбрано точек:",
    len(st.session_state.points),
    "/ 4"
)

for index, point in enumerate(
    st.session_state.points
):

    st.write(
        index + 1,
        ":",
        point
    )


# ============================================================
# СБРОС
# ============================================================

if st.button("🔄 Сбросить выбор"):

    st.session_state.points = []
    st.session_state.last_click = None

    st.rerun()


if len(st.session_state.points) != 4:

    st.info(
        "Нужно выбрать ровно 4 точки."
    )

    st.stop()


# ============================================================
# УПОРЯДОЧИВАЕМ ТОЧКИ
# ============================================================

points = np.array(
    st.session_state.points,
    dtype=np.float32
)

sums = points.sum(axis=1)
diffs = np.diff(
    points,
    axis=1
).reshape(-1)

ordered = np.zeros(
    (4, 2),
    dtype=np.float32
)

ordered[0] = points[np.argmin(sums)]
ordered[2] = points[np.argmax(sums)]
ordered[1] = points[np.argmin(diffs)]
ordered[3] = points[np.argmax(diffs)]


# ============================================================
# ПЕРСПЕКТИВА
# ============================================================

top_width = np.linalg.norm(
    ordered[1] - ordered[0]
)

bottom_width = np.linalg.norm(
    ordered[2] - ordered[3]
)

left_height = np.linalg.norm(
    ordered[3] - ordered[0]
)

right_height = np.linalg.norm(
    ordered[2] - ordered[1]
)

crop_width = max(
    100,
    int(max(top_width, bottom_width))
)

crop_height = max(
    100,
    int(max(left_height, right_height))
)

destination = np.array(
    [
        [0, 0],
        [crop_width - 1, 0],
        [crop_width - 1, crop_height - 1],
        [0, crop_height - 1]
    ],
    dtype=np.float32
)

matrix = cv2.getPerspectiveTransform(
    ordered,
    destination
)

crop = cv2.warpPerspective(
    image,
    matrix,
    (
        crop_width,
        crop_height
    )
)


# ============================================================
# ПОКАЗ ОБЛАСТИ
# ============================================================

st.header("2. Выбранный участок")

st.image(
    crop,
    use_container_width=True
)


# ============================================================
# СОЗДАНИЕ
# ============================================================

create = st.button(
    "🧿 СОЗДАТЬ 3D МОДЕЛЬ",
    type="primary",
    use_container_width=True
)


if not create:
    st.stop()


# ============================================================
# COLOR ANALYSIS
# ============================================================

with st.spinner(
    "Распознаю цвета и создаю 3D..."
):

    processing_width = min(
        600,
        crop.shape[1]
    )

    processing_height = int(
        crop.shape[0]
        * processing_width
        / crop.shape[1]
    )

    processing = cv2.resize(
        crop,
        (
            processing_width,
            processing_height
        ),
        interpolation=cv2.INTER_AREA
    )

    lab = cv2.cvtColor(
        processing,
        cv2.COLOR_RGB2LAB
    )

    pixels = lab.reshape(
        -1,
        3
    ).astype(
        np.float32
    )


    # ========================================================
    # K-MEANS
    # ========================================================

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        color_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(
        processing_height,
        processing_width
    )


    # ========================================================
    # НАСТОЯЩИЕ РАЗМЕРЫ МОДЕЛИ
    # ========================================================

    model_width = 1.0

    model_height = (
        model_width
        * crop.shape[0]
        / crop.shape[1]
    )

    scale_x = (
        model_width
        / processing_width
    )

    scale_y = (
        model_height
        / processing_height
    )

    base_z = (
        carpet_thickness_mm
        / 1000.0
    )

    ornament_z = (
        ornament_height_mm
        / 1000.0
    )


    # ========================================================
    # ВЕРШИНЫ И ГРАНИ
    # ========================================================

    vertices = []
    faces = []
    face_materials = []


    # ========================================================
    # ОСНОВА КОВРА
    # ========================================================

    base_vertices = [
        (
            -model_width / 2,
            -model_height / 2,
            0
        ),
        (
            model_width / 2,
            -model_height / 2,
            0
        ),
        (
            model_width / 2,
            model_height / 2,
            0
        ),
        (
            -model_width / 2,
            model_height / 2,
            0
        ),
        (
            -model_width / 2,
            -model_height / 2,
            base_z
        ),
        (
            model_width / 2,
            -model_height / 2,
            base_z
        ),
        (
            model_width / 2,
            model_height / 2,
            base_z
        ),
        (
            -model_width / 2,
            model_height / 2,
            base_z
        )
    ]

    base_start = len(vertices)

    vertices.extend(
        base_vertices
    )

    base_faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0)
    ]

    for face in base_faces:

        faces.append(
            tuple(
                base_start + index
                for index in face
            )
        )

        face_materials.append(
            0
        )


    # ========================================================
    # ПЛАН МАТЕРИАЛОВ
    # ========================================================

    materials = [
        (
            "Carpet_Base",
            (110, 80, 50)
        )
    ]


    # ========================================================
    # ПЛОЩАДЬ ЦВЕТОВ
    # ========================================================

    cluster_areas = []

    for cluster_id in range(
        color_count
    ):

        mask = np.uint8(
            labels == cluster_id
        ) * 255

        area = int(
            np.count_nonzero(mask)
        )

        cluster_areas.append(
            area
        )

    largest_cluster = int(
        np.argmax(cluster_areas)
    )


    # ========================================================
    # ЦВЕТОВЫЕ ЭЛЕМЕНТЫ
    # ========================================================

    element_count = 0


    for cluster_id in range(
        color_count
    ):

        if (
            ignore_background
            and
            cluster_id == largest_cluster
        ):
            continue


        mask = np.uint8(
            labels == cluster_id
        ) * 255


        # Убираем шум
        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # ====================================================
        # КОНТУРЫ
        # ====================================================

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # ====================================================
        # ЦВЕТ
        # ====================================================

        center = centers[
            cluster_id
        ].reshape(
            1,
            1,
            3
        ).astype(
            np.uint8
        )

        rgb = cv2.cvtColor(
            center,
            cv2.COLOR_LAB2RGB
        )[0, 0]

        color = (
            int(rgb[0]),
            int(rgb[1]),
            int(rgb[2])
        )


        material_id = len(materials)

        materials.append(
            (
                "Color_" + str(material_id),
                color
            )
        )


        # ====================================================
        # КОНТУРЫ → 3D
        # ====================================================

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < min_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            epsilon = (
                0.004
                * perimeter
            )

            polygon = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            if len(polygon) < 3:
                continue


            # =================================================
            # ПОЛИГОН
            # =================================================

            poly_points = []

            for point in polygon:

                px = float(
                    point[0][0]
                )

                py = float(
                    point[0][1]
                )

                x = (
                    px * scale_x
                    - model_width / 2
                )

                y = (
                    model_height
                    - py * scale_y
                    - model_height / 2
                )

                poly_points.append(
                    (x, y)
                )


            if len(poly_points) < 3:
                continue


            # =================================================
            # ПРОСТОЕ TRIANGULATION
            # =================================================

            center_x = sum(
                point[0]
                for point in poly_points
            ) / len(poly_points)

            center_y = sum(
                point[1]
                for point in poly_points
            ) / len(poly_points)


            center_index = len(vertices)

            vertices.append(
                (
                    center_x,
                    center_y,
                    base_z
                )
            )

            top_center_index = len(vertices)

            vertices.append(
                (
                    center_x,
                    center_y,
                    base_z + ornament_z
                )
            )


            # Добавляем вершины контура
            bottom_indices = []
            top_indices = []


            for x, y in poly_points:

                bottom_indices.append(
                    len(vertices)
                )

                vertices.append(
                    (
                        x,
                        y,
                        base_z
                    )
                )

                top_indices.append(
                    len(vertices)
                )

                vertices.append(
                    (
                        x,
                        y,
                        base_z + ornament_z
                    )
                )


            count = len(
                poly_points
            )


            # =================================================
            # ВЕРХ
            # =================================================

            for i in range(count):

                j = (
                    i + 1
                ) % count

                faces.append(
                    (
                        top_center_index,
                        top_indices[i],
                        top_indices[j]
                    )
                )

                face_materials.append(
                    material_id
                )


            # =================================================
            # НИЗ
            # =================================================

            for i in range(count):

                j = (
                    i + 1
                ) % count

                faces.append(
                    (
                        center_index,
                        bottom_indices[j],
                        bottom_indices[i]
                    )
                )

                face_materials.append(
                    material_id
                )


            # =================================================
            # БОКОВЫЕ СТЕНКИ
            # =================================================

            for i in range(count):

                j = (
                    i + 1
                ) % count

                faces.append(
                    (
                        bottom_indices[i],
                        bottom_indices[j],
                        top_indices[j]
                    )
                )

                face_materials.append(
                    material_id
                )

                faces.append(
                    (
                        bottom_indices[i],
                        top_indices[j],
                        top_indices[i]
                    )
                )

                face_materials.append(
                    material_id
                )


            element_count += 1


# ============================================================
# РЕЗУЛЬТАТ
# ============================================================

st.header("3. Готовая 3D модель")

st.success(
    "Создано элементов орнамента: "
    + str(element_count)
)


# ============================================================
# PLOTLY PREVIEW
# ============================================================

vertices_array = np.array(
    vertices,
    dtype=np.float32
)

faces_array = np.array(
    faces,
    dtype=np.int32
)


fig = go.Figure()


# Используем цвет материала
for material_id in range(
    len(materials)
):

    selected = [
        index
        for index, value
        in enumerate(face_materials)
        if value == material_id
    ]

    if not selected:
        continue

    selected_faces = faces_array[
        selected
    ]

    color = materials[
        material_id
    ][1]

    fig.add_trace(
        go.Mesh3d(
            x=vertices_array[:, 0],
            y=vertices_array[:, 1],
            z=vertices_array[:, 2],
            i=selected_faces[:, 0],
            j=selected_faces[:, 1],
            k=selected_faces[:, 2],
            color=(
                "rgb("
                + str(color[0])
                + ","
                + str(color[1])
                + ","
                + str(color[2])
                + ")"
            ),
            flatshading=True,
            opacity=1.0
        )
    )


fig.update_layout(
    height=650,
    margin=dict(
        l=0,
        r=0,
        t=20,
        b=0
    ),
    scene=dict(
        aspectmode="data"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ЭКСПОРТ OBJ + MTL
# ============================================================

st.header("4. Экспорт для 3ds Max")

export_dir = tempfile.mkdtemp()

obj_file = os.path.join(
    export_dir,
    "Tamga3D_model.obj"
)

mtl_file = os.path.join(
    export_dir,
    "Tamga3D_model.mtl"
)

texture_file = os.path.join(
    export_dir,
    "Tamga3D_texture.png"
)

zip_file = os.path.join(
    export_dir,
    "Tamga3D_3dsMax.zip"
)


# ============================================================
# OBJ
# ============================================================

with open(
    obj_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "mtllib Tamga3D_model.mtl\n"
    )

    file.write(
        "o Tamga3D_Carpet\n"
    )


    for vertex in vertices:

        file.write(
            "v "
            + str(vertex[0])
            + " "
          
