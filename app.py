import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon, MultiPolygon
from trimesh.creation import extrude_polygon
import trimesh

from streamlit_image_coordinates import streamlit_image_coordinates

import tempfile
import os
import zipfile
from PIL import Image


# ============================================================
# TAMGA3D — PROTOTYPE
# ============================================================

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧿",
    layout="wide"
)


# ============================================================
# СТИЛЬ
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.subtitle {
    color: #777;
    font-size: 18px;
    margin-bottom: 25px;
}

.step {
    font-size: 24px;
    font-weight: 600;
    margin-top: 25px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">🧿 Tamga3D</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '2D орнамент → цвет → объёмный 3D-модельный элемент → OBJ для 3ds Max'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

MAX_IMAGE_SIZE = 900

# Толщина настоящего ковра
BASE_THICKNESS = 0.004

# Высота орнамента над ковром
ORNAMENT_HEIGHT = 0.0025

# Количество цветов
DEFAULT_COLORS = 6

# Минимальный размер элемента
DEFAULT_MIN_AREA = 120


# ============================================================
# SESSION STATE
# ============================================================

if "points" not in st.session_state:
    st.session_state.points = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

if "model_data" not in st.session_state:
    st.session_state.model_data = None


# ============================================================
# ЗАГРУЗКА ФОТО
# ============================================================

uploaded = st.file_uploader(
    "Загрузите фотографию ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:

    st.info(
        "Загрузите фотографию ковра, чтобы начать."
    )

    st.stop()


# ============================================================
# ОТКРЫВАЕМ ФОТО
# ============================================================

data = np.asarray(
    bytearray(uploaded.read()),
    dtype=np.uint8
)

img = cv2.imdecode(
    data,
    cv2.IMREAD_COLOR
)

if img is None:

    st.error(
        "Не удалось открыть изображение."
    )

    st.stop()


img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# УМЕНЬШАЕМ ФОТО
# ============================================================

original_h, original_w = img.shape[:2]

resize_factor = min(
    1.0,
    MAX_IMAGE_SIZE / max(
        original_h,
        original_w
    )
)

if resize_factor < 1:

    img = cv2.resize(
        img,
        (
            int(original_w * resize_factor),
            int(original_h * resize_factor)
        ),
        interpolation=cv2.INTER_AREA
    )


img_h, img_w = img.shape[:2]


# ============================================================
# НАСТРОЙКИ СПРАВА
# ============================================================

with st.sidebar:

    st.header("⚙️ Настройки 3D")

    color_count = st.slider(
        "Количество цветов",
        min_value=2,
        max_value=10,
        value=DEFAULT_COLORS
    )

    min_area = st.slider(
        "Минимальный размер элемента",
        min_value=30,
        max_value=1000,
        value=DEFAULT_MIN_AREA,
        step=10
    )

    base_thickness_mm = st.slider(
        "Толщина ковра, мм",
        min_value=1.0,
        max_value=8.0,
        value=4.0,
        step=0.5
    )

    ornament_height_mm = st.slider(
        "Высота орнамента, мм",
        min_value=0.5,
        max_value=6.0,
        value=2.5,
        step=0.5
    )

    ignore_largest = st.checkbox(
        "Не поднимать самый большой цвет",
        value=False,
        help=(
            "Полезно, если выбранная область содержит "
            "фон ковра и орнамент."
        )
    )


# ============================================================
# ШАГ 1
# ============================================================

st.markdown(
    '<div class="step">1. Выберите участок орнамента</div>',
    unsafe_allow_html=True
)

st.write(
    "Нажмите четыре точки вокруг участка, "
    "который хотите превратить в 3D."
)

st.image(
    img,
    use_container_width=True
)


# ============================================================
# КЛИК ПО ИЗОБРАЖЕНИЮ
# ============================================================

clicked = streamlit_image_coordinates(
    img,
    key="tamga3d_selector"
)

if clicked is not None:

    point = (
        int(clicked["x"]),
        int(clicked["y"])
    )

    if point != st.session_state.last_click:

        if len(st.session_state.points) < 4:

            st.session_state.points.append(
                point
            )

            st.session_state.last_click = point

            st.rerun()


# ============================================================
# ТОЧКИ
# ============================================================

if st.session_state.points:

    st.write(
        f"Выбрано: "
        f"{len(st.session_state.points)} / 4"
    )

    cols = st.columns(4)

    for i, point in enumerate(
        st.session_state.points
    ):

        cols[i].write(
            f"**{i + 1}**  "
            f"{point[0]}, {point[1]}"
        )


# ============================================================
# СБРОС
# ============================================================

if st.button(
    "🔄 Сбросить область"
):

    st.session_state.points = []
    st.session_state.last_click = None
    st.session_state.model_data = None

    st.rerun()


if len(st.session_state.points) < 4:

    st.info(
        "Выберите четыре точки вокруг нужного участка."
    )

    st.stop()


# ============================================================
# СОРТИРОВКА 4 ТОЧЕК
# ============================================================

pts = np.array(
    st.session_state.points,
    dtype=np.float32
)

s = pts.sum(axis=1)

d = np.diff(
    pts,
    axis=1
).reshape(-1)

ordered = np.zeros(
    (4, 2),
    dtype=np.float32
)

ordered[0] = pts[np.argmin(s)]
ordered[2] = pts[np.argmax(s)]
ordered[1] = pts[np.argmin(d)]
ordered[3] = pts[np.argmax(d)]


# ============================================================
# CROP / PERSPECTIVE
# ============================================================

width1 = np.linalg.norm(
    ordered[1] - ordered[0]
)

width2 = np.linalg.norm(
    ordered[2] - ordered[3]
)

height1 = np.linalg.norm(
    ordered[3] - ordered[0]
)

height2 = np.linalg.norm(
    ordered[2] - ordered[1]
)

crop_w = int(
    max(
        width1,
        width2
    )
)

crop_h = int(
    max(
        height1,
        height2
    )
)

crop_w = max(
    crop_w,
    100
)

crop_h = max(
    crop_h,
    100
)


destination = np.array(
    [
        [0, 0],
        [crop_w - 1, 0],
        [crop_w - 1, crop_h - 1],
        [0, crop_h - 1]
    ],
    dtype=np.float32
)


matrix = cv2.getPerspectiveTransform(
    ordered,
    destination
)

crop = cv2.warpPerspective(
    img,
    matrix,
    (
        crop_w,
        crop_h
    )
)


# ============================================================
# ПОКАЗ ВЫБРАННОГО УЧАСТКА
# ============================================================

st.markdown(
    '<div class="step">2. Выбранный участок</div>',
    unsafe_allow_html=True
)

st.image(
    crop,
    use_container_width=True
)


# ============================================================
# СОЗДАНИЕ 3D
# ============================================================

create_button = st.button(
    "🧿 СОЗДАТЬ 3D МОДЕЛЬ",
    type="primary",
    use_container_width=True
)


if create_button:

    with st.spinner(
        "Tamga3D распознаёт цвета и создаёт геометрию..."
    ):

        # ====================================================
        # НОРМАЛИЗАЦИЯ РАЗМЕРА
        # ====================================================

        processing_w = min(
            crop_w,
            700
        )

        processing_h = int(
            crop_h *
            processing_w /
            crop_w
        )

        processing = cv2.resize(
            crop,
            (
                processing_w,
                processing_h
            ),
            interpolation=cv2.INTER_AREA
        )


        # ====================================================
        # LAB COLOR SPACE
        # ====================================================

        lab = cv2.cvtColor(
            processing,
            cv2.COLOR_RGB2LAB
        )

        pixels = lab.reshape(
            (-1, 3)
        ).astype(
            np.float32
        )


        # ====================================================
        # K-MEANS
        # ====================================================

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.5
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            color_count,
            None,
            criteria,
            8,
            cv2.KMEANS_PP_CENTERS
        )

        labels = labels.reshape(
            processing_h,
            processing_w
        )


        # ====================================================
        # РАЗМЕР 3D
        # ====================================================

        MODEL_WIDTH = 1.0

        MODEL_HEIGHT = (
            MODEL_WIDTH *
            crop_h /
            crop_w
        )

        scale_x = (
            MODEL_WIDTH /
            processing_w
        )

        scale_y = (
            MODEL_HEIGHT /
            processing_h
        )


        # ====================================================
        # ОСНОВА КОВРА
        # ====================================================

        base_thickness = (
            base_thickness_mm / 1000
        )

        ornament_height = (
            ornament_height_mm / 1000
        )

        base = trimesh.creation.box(
            extents=[
                MODEL_WIDTH,
                MODEL_HEIGHT,
                base_thickness
            ]
        )

        base.apply_translation(
            [
                0,
                0,
                base_thickness / 2
            ]
        )


        # ====================================================
        # ЦВЕТА КЛАСТЕРОВ
        # ====================================================

        cluster_info = []

        for cluster_id in range(
            color_count
        ):

            mask = np.uint8(
                labels == cluster_id
            ) * 255

            area = np.count_nonzero(
                mask
            )

            # Lab → RGB
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

            cluster_info.append(
                {
                    "id": cluster_id,
                    "area": int(area),
                    "rgb": (
                        int(rgb[0]),
                        int(rgb[1]),
                        int(rgb[2])
                    )
                }
            )


        # ====================================================
        # САМЫЙ БОЛЬШОЙ ЦВЕТ
        # ====================================================

        largest_cluster = max(
            cluster_info,
            key=lambda x: x["area"]
        )["id"]


        # ====================================================
        # 3D ЭЛЕМЕНТЫ
        # ====================================================

        pieces = []


        for info in cluster_info:

            cluster_id = info["id"]

            if (
                ignore_largest
                and
                cluster_id == largest_cluster
            ):
                continue


            mask = np.uint8(
                labels == cluster_id
            ) * 255


            # =================================================
            # УДАЛЕНИЕ ШУМА
            # =================================================

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


            # =================================================
            # КОНТУРЫ
            # =================================================

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


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
                    0.004 *
                    perimeter
                )

                approx = cv2.approxPolyDP(
                    contour,
                    epsilon,
                    True
                )


                if len(approx) < 3:
                    continue


                polygon_points = []


                for p in approx:

                    px = float(
                        p[0][0]
                    )

                    py = float(
                        p[0][1]
                    )


                    x = (
                        px *
                        scale_x
                    )

                    y = (
                        MODEL_HEIGHT -
                        py * scale_y
                    )


                    x -= MODEL_WIDTH / 2
                    y -= MODEL_HEIGHT / 2


                    polygon_points.append(
                        (
                            x,
                            y
                        )
                    )


                try:

                    polygon = Polygon(
                        polygon_points
                    )


                    if not polygon.is_valid:

                        polygon = polygon.buffer(
                            0
                        )


                    if polygon.is_empty:
                        continue


                    polygons = []


                    if isinstance(
                        polygon,
                        Polygon
                    ):

                        polygons = [
                            polygon
                        ]

                    elif isinstance(
                        polygon,
                        MultiPolygon
                    ):

                        polygons = list(
                            polygon.geoms
                        )


                    for poly in polygons:

                        if poly.area <= 0:
                            continue


                        mesh = extrude_polygon(
                            poly,
                            height=ornament_height
                        )


                        mesh.apply_translation(
                            [
                                0,
                                0,
                                base_thickness
                            ]
                        )


                        color = info["rgb"]


                        mesh.visual.face_colors = np.tile(
                            np.array(
                                [
                                    color[0],
                                    color[1],
                                    color[2],
                                    255
                                ],
                                dtype=np.uint8
                            ),
                            (
                                len(mesh.faces),
                                1
                            )
                        )


                        pieces.append(
                            {
                                "mesh": mesh,
                                "color": color
                            }
                        )


                except Exception:
                    continue


        # ====================================================
        # РЕЗУЛЬТАТ
        # ====================================================

        if not pieces:

            st.error(
                "Не удалось обнаружить элементы орнамента. "
                "Попробуйте увеличить выбранную область "
                "или уменьшить минимальный размер элемента."
            )

            st.stop()


        # ====================================================
        # СОХРАНЯЕМ
        # ====================================================

        st.session_state.model_data = {
            "base": base,
            "pieces": pieces,
            "crop": crop,
            "model_width": MODEL_WIDTH,
            "model_height": MODEL_HEIGHT,
            "base_thickness": base_thickness,
            "ornament_height": ornament_height
        }


# ============================================================
# ПОКАЗ РЕЗУЛЬТАТА
# ============================================================

if st.session_state.model_data is None:

    st.stop()


data_model = st.session_state.model_data

base = data_model["base"]
pieces = data_model["pieces"]


# ============================================================
# СТАТИСТИКА
# ============================================================

st.markdown(
    '<div class="step">3. Результат</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Цветовых элементов",
    len(pieces)
)

c2.metric(
    "Толщина основы",
    f"{base_thickness_mm:.1f} мм"
)

c3.metric(
    "Высота орнамента",
    f"{ornament_height_mm:.1f} мм"
)


# ============================================================
# 3D PREVIEW
# ============================================================

st.markdown(
    '<div class="step">4. 3D-просмотр</div>',
    unsafe_allow_html=True
)


fig = go.Figure()


# ============================================================
# ОСНОВА
# ============================================================

v = base.vertices
f = base.faces

fig.add_trace(
    go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color="rgb(120,120,120)",
        flatshading=True,
        opacity=1
    )
)


# ============================================================
# ОРНАМЕНТ
# ============================================================

for piece in pieces:

    mesh = piece["mesh"]
    color = piece["color"]

    v = mesh.vertices
    f = mesh.faces

    fig.add_trace(
        go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color=(
                f"rgb("
                f"{color[0]},"
                f"{color[1]},"
                f"{
