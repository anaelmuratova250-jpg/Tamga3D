import io
import zipfile
import math
import random

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================================================
# TAMGA3D
# =========================================================

st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Tamga3D")
st.caption("Фото ковра → 4-точечный выбор → цветной объёмный 3D-ковёр")


# =========================================================
# НАСТРОЙКИ
# =========================================================

st.sidebar.header("⚙️ Настройки")

COLORS = st.sidebar.slider(
    "Количество цветов",
    2,
    8,
    5
)

MIN_AREA = st.sidebar.slider(
    "Минимальный размер деталей",
    20,
    1000,
    100
)

CARPET_THICKNESS = st.sidebar.slider(
    "Толщина ковра, мм",
    1.0,
    8.0,
    3.0,
    0.5
)

ORNAMENT_HEIGHT = st.sidebar.slider(
    "Высота орнамента, мм",
    0.5,
    8.0,
    3.0,
    0.5
)

WOOL = st.sidebar.checkbox(
    "🧶 Добавить шерстяной ворс",
    True
)

WOOL_DENSITY = st.sidebar.slider(
    "Плотность шерсти",
    1,
    5,
    2
)

WOOL_LENGTH = st.sidebar.slider(
    "Длина волокна, мм",
    0.5,
    3.0,
    1.2,
    0.1
)

st.sidebar.info(
    "Для выбора используй 4 клика: "
    "левый верхний → правый верхний → "
    "правый нижний → левый нижний."
)


# =========================================================
# 4-ТОЧЕЧНАЯ ПЕРСПЕКТИВА
# =========================================================

def order_points(points):
    pts = np.array(points, dtype=np.float32)

    result = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    result[0] = pts[np.argmin(s)]
    result[2] = pts[np.argmax(s)]
    result[1] = pts[np.argmin(d)]
    result[3] = pts[np.argmax(d)]

    return result


def perspective_crop(image, points):
    pts = order_points(points)

    tl, tr, br, bl = pts

    width1 = np.linalg.norm(tr - tl)
    width2 = np.linalg.norm(br - bl)

    height1 = np.linalg.norm(bl - tl)
    height2 = np.linalg.norm(br - tr)

    width = int(max(width1, width2))
    height = int(max(height1, height2))

    width = max(width, 50)
    height = max(height, 50)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        pts,
        destination
    )

    result = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return result


# =========================================================
# РАСПОЗНАВАНИЕ ЦВЕТОВ
# =========================================================

def quantize_colors(image, number):
    h, w = image.shape[:2]

    small = cv2.resize(
        image,
        (min(160, w), min(160, h))
    )

    lab = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2LAB
    )

    data = lab.reshape(
        (-1, 3)
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        data,
        number,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    quantized_lab = centers[
        labels.flatten()
    ].reshape(lab.shape)

    quantized_rgb = cv2.cvtColor(
        quantized_lab,
        cv2.COLOR_LAB2RGB
    )

    quantized_rgb = cv2.resize(
        quantized_rgb,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return quantized_rgb, centers


def create_masks(image, number):
    quantized, lab_centers = quantize_colors(
        image,
        number
    )

    rgb_centers = cv2.cvtColor(
        lab_centers.reshape(1, -1, 3),
        cv2.COLOR_LAB2RGB
    )[0]

    masks = []

    for color in rgb_centers:

        distance = np.linalg.norm(
            quantized.astype(np.float32)
            - color.astype(np.float32),
            axis=2
        )

        mask = np.where(
            distance < 10,
            255,
            0
        ).astype(np.uint8)

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

        masks.append(mask)

    return masks, rgb_centers


# =========================================================
# КОНТУРЫ
# =========================================================

def get_polygons(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

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

        epsilon = max(
            0.002 * perimeter,
            0.5
        )

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(polygon) >= 3:

            polygon = polygon.reshape(
                (-1, 2)
            )

            polygons.append(
                polygon
            )

    return polygons


# =========================================================
# ПРОСТАЯ ТРИАНГУЛЯЦИЯ
# =========================================================

def triangle_area(a, b, c):
    return (
        (b[0] - a[0])
        * (c[1] - a[1])
        -
        (b[1] - a[1])
        * (c[0] - a[0])
    )


def point_in_triangle(p, a, b, c):
    d1 = triangle_area(p, a, b)
    d2 = triangle_area(p, b, c)
    d3 = triangle_area(p, c, a)

    has_negative = (
        d1 < 0 or
        d2 < 0 or
        d3 < 0
    )

    has_positive = (
        d1 > 0 or
        d2 > 0 or
        d3 > 0
    )

    return not (
        has_negative and has_positive
    )


def triangulate(points):
    pts = [
        tuple(map(float, p))
        for p in points
    ]

    if len(pts) < 3:
        return []

    area = 0

    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]

        area += (
            x1 * y2 -
            x2 * y1
        )

    if area < 0:
        pts.reverse()

    remaining = list(
        range(len(pts))
    )

    triangles = []

    guard = 0

    while len(remaining) > 3:
        guard += 1

        if guard > 10000:
            break

        found = False

        for i in range(
            len(remaining)
        ):

            a_index = remaining[i - 1]
            b_index = remaining[i]
            c_index = remaining[
                (i + 1) % len(remaining)
            ]

            a = pts[a_index]
            b = pts[b_index]
            c = pts[c_index]

            if triangle_area(
                a,
                b,
                c
            ) <= 0:
                continue

            inside = False

            for other in remaining:

                if other in (
                    a_index,
                    b_index,
                    c_index
                ):
                    continue

                if point_in_triangle(
                    pts[other],
                    a,
                    b,
                    c
                ):
                    inside = True
                    break

            if inside:
                continue

            triangles.append(
                (
                    a_index,
                    b_index,
                    c_index
                )
            )

            remaining.pop(i)

            found = True
            break

        if not found:
            break

    if len(remaining) == 3:
        triangles.append(
            (
                remaining[0],
                remaining[1],
                remaining[2]
            )
        )

    return triangles


# =========================================================
# 3D MODEL
# =========================================================

class Model:

    def __init__(self):
        self.vertices = []
        self.faces = []

    def vertex(self, x, y, z):
        self.vertices.append(
            (
                float(x),
                float(y),
                float(z)
            )
        )

        return len(self.vertices)

    def face(
        self,
        a,
        b,
        c,
        material
    ):
        self.faces.append(
            (
                a,
                b,
                c,
                material
            )
        )


def add_box(
    model,
    x1,
    y1,
    x2,
    y2,
    z1,
    z2,
    material
):

    a = model.vertex(
        x1, y1, z1
    )
    b = model.vertex(
        x2, y1, z1
    )
    c = model.vertex(
        x2, y2, z1
    )
    d = model.vertex(
        x1, y2, z1
    )

    e = model.vertex(
        x1, y1, z2
    )
    f = model.vertex(
        x2, y1, z2
    )
    g = model.vertex(
        x2, y2, z2
    )
    h = model.vertex(
        x1, y2, z2
    )

    model.face(a, c, b, material)
    model.face(a, d, c, material)

    model.face(e, f, g, material)
    model.face(e, g, h, material)

    model.face(a, b, f, material)
    model.face(a, f, e, material)

    model.face(b, c, g, material)
    model.face(b, g, f, material)

    model.face(c, d, h, material)
    model.face(c, h, g, material)

    model.face(d, a, e, material)
    model.face(d, e, h, material)


def add_prism(
    model,
    polygon,
    image_width,
    image_height,
    bottom_z,
    top_z,
    material
):

    polygon = np.asarray(
        polygon,
        dtype=np.float32
    )

    triangles = triangulate(
        polygon
    )

    if not triangles:
        return

    bottom = []
    top = []

    for x, y in polygon:

        px = (
            x / image_width
            - 0.5
        )

        py = (
            0.5
            - y / image_height
        )

        bottom.append(
            model.vertex(
                px,
                py,
                bottom_z
            )
        )

        top.append(
            model.vertex(
                px,
                py,
                top_z
            )
        )

    for a, b, c in triangles:

        model.face(
            top[a],
            top[b],
            top[c],
            material
        )

        model.face(
            bottom[c],
            bottom[b],
            bottom[a],
            material
        )

    count = len(
        polygon
    )

    for i in range(count):

        j = (
            i + 1
        ) % count

        model.face(
            bottom[i],
            bottom[j],
            top[j],
            material
        )

        model.face(
            bottom[i],
            top[j],
            top[i],
            material
        )


# =========================================================
# ШЕРСТЯНОЙ ВОРС
# =========================================================

def add_wool_tuft(
    model,
    x,
    y,
    z,
    length,
    material,
    angle
):

    radius = 0.0012

    dx = math.cos(angle)
    dy = math.sin(angle)

    x2 = x + dx * length
    y2 = y + dy * length

    # маленький четырёхугольный
    # "пучок" шерсти

    v1 = model.vertex(
        x - radius,
        y - radius,
        z
    )

    v2 = model.vertex(
        x + radius,
        y + radius,
        z
    )

    v3 = model.vertex(
        x2 + radius * 0.5,
        y2 + radius * 0.5,
        z + length * 0.4
    )

    v4 = model.vertex(
        x2 - radius * 0.5,
        y2 - radius * 0.5,
        z + length * 0.4
    )

    model.face(
        v1,
        v2,
        v3,
        material
    )

    model.face(
        v1,
        v3,
        v4,
        material
    )


def add_wool(
    model,
    image,
    masks,
    image_width,
    image_height,
    base_z,
    ornament_z,
    density,
    length_mm,
    colors
):

    random.seed(12)

    # масштаб мм → метры
    length = length_mm / 1000.0

    # Ограничиваем количество ворсинок,
    # чтобы OBJ не стал огромным.
    step = max(
        3,
        8 - density
    )

    for material, mask in enumerate(
        masks
    ):

        ys, xs = np.where(
            mask > 0
        )

        if len(xs) == 0:
            continue

        indices = np.arange(
            0,
            len(xs),
            step
        )

        for index in indices:

            px = xs[index]
            py = ys[index]

            x = (
                px / image_width
                - 0.5
            )

            y = (
                0.5
                - py / image_height
            )

            # Небольшая случайность
            # делает ворс менее искусственным.
            jitter_x = (
                random.random()
                - 0.5
            ) * 0.004

            jitter_y = (
                random.random()
                - 0.5
            ) * 0.004

            angle = random.random() * (
                math.pi * 2
            )

            length_random = (
                length
                * (
                    0.65
                    + random.random() * 0.7
                )
            )

            add_wool_tuft(
                model,
                x + jitter_x,
                y + jitter_y,
                base_z + ornament_z,
                length_random,
                material,
                angle
            )


# =========================================================
# OBJ
# =========================================================

def make_obj(model):

    lines = []

    lines.append(
        "mtllib Tamga3D_model.mtl"
    )

    lines.append(
        "o Tamga3D_Carpet"
    )

    for vertex in model.vertices:

        x, y, z = vertex

        lines.append(
            f"v {x:.6f} {y:.6f} {z:.6f}"
        )

    current_material = None

    for face in model.faces:

        a, b, c, material = face

        if material != current_material:

            lines.append(
                f"usemtl material_{material}"
            )

            current_material = material

        lines.append(
            f"f {a} {b} {c}"
        )

    return "\n".join(
        lines
    )


def make_mtl(colors):

    lines = []

    for i, color in enumerate(
        colors
    ):

        r = color[0] / 255
        g = color[1] / 255
        b = color[2] / 255

        lines.append(
            f"newmtl material_{i}"
        )

        lines.append(
            f"Kd {r:.4f} {g:.4f} {b:.4f}"
        )

        lines.append(
            "Ka 0.08 0.08 0.08"
        )

        lines.append(
            "Ks 0.05 0.05 0.05"
        )

        lines.append(
            "Ns 10"
        )

        lines.append("")

    return "\n".join(
        lines
    )


def create_zip(
    obj_text,
    mtl_text,
    image
):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "Tamga3D_model.obj",
            obj_text
        )

        archive.writestr(
            "Tamga3D_model.mtl",
            mtl_text
        )

        image_buffer = io.BytesIO()

        Image.fromarray(
            image
        ).save(
            image_buffer,
            format="PNG"
        )

        archive.writestr(
            "Tamga3D_texture.png",
            image_buffer.getvalue()
        )

    buffer.seek(0)

    return buffer


# =========================================================
# ЗАГРУЗКА
# =========================================================

uploaded = st.file_uploader(
    "📷 Загрузи фотографию ковра",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

if uploaded is None:

    st.info(
        "Загрузи фотографию ковра, "
        "чтобы начать."
    )

    st.stop()


image = np.array(
    Image.open(
        uploaded
    ).convert("RGB")
)


# =========================================================
# ВЫБОР 4 ТОЧЕК
# =========================================================

st.subheader(
    "1. Выбери 4 угла области"
)

max_width = 800

scale = min(
    1.0,
    max_width / image.shape[1]
)

display_width = int(
    image.shape[1] * scale
)

display_height = int(
    image.shape[0] * scale
)

display_image = cv2.resize(
    image,
    (
        display_width,
        display_height
    )
)

coords = streamlit_image_coordinates(
    Image.fromarray(
        display_image
    ),
    key="four_point_selector"
)


if "points" not in st.session_state:
    st.session_state.points = []


if coords:

    point = (
        int(coords["x"] / scale),
        int(coords["y"] / scale)
    )

    last = (
        st.session_state.points[-1]
        if st.session_state.points
        else None
    )

    if point != last:

        st.session_state.points.append(
            point
        )

    if len(
        st.session_state.points
    ) > 4:

        st.session_state.points = (
            st.session_state.points[-4:]
        )


st.write(
    f"Выбрано точек: "
    f"{len(st.session_state.points)} / 4"
)


if st.session_state.points:

    st.write(
        st.session_state.points
    )


if st.button(
    "🗑️ Очистить точки"
):

    st.session_state.points = []

    st.session_state.pop(
        "selected",
        None
    )

    st.session_state.pop(
        "model",
        None
    )

    st.rerun()


# =========================================================
# ПЕРСПЕКТИВНЫЙ CROP
# =========================================================

if len(
    st.session_state.points
) == 4:

    selected = perspective_crop(
        image,
        st.session_state.points
    )

    st.session_state.selected = (
        selected
    )

    st.success(
        "4 точки выбраны. "
        "Перспектива исправлена."
    )

    st.image(
        selected,
        caption="Выбранная область ковра",
        use_container_width=True
    )

else:

    st.warning(
        "Нужно поставить ровно 4 точки."
    )

    st.stop()


# =========================================================
# ЦВЕТА
# =========================================================

st.subheader(
    "2. Распознавание цветов"
)

selected = st.session_state.selected

masks, color_centers = create_masks(
    selected,
    COLORS
)

preview = np.zeros_like(
    selected
)

for mask, color in zip(
    masks,
    color_centers
):

    preview[
        mask > 0
    ] = color


st.image(
    preview,
    caption="Сегментация орнамента",
    use_container_width=True
)


# =========================================================
# СОЗДАНИЕ
# =========================================================

st.subheader(
    "3. Создание 3D"
)

if st.button(
    "🧶 Создать 3D-модель",
    type="primary"
):

    h, w = selected.shape[:2]

    model = Model()

    carpet_z = (
        CARPET_THICKNESS / 1000
    )

    ornament_z = (
        ORNAMENT_HEIGHT / 1000
    )

    # Основа ковра.
    add_box(
        model,
        -0.5,
        -0.5,
        0.5,
        0.5,
        0,
        carpet_z,
        0
    )

    details = 0

    for material, mask in enumerate(
        masks
    ):

        polygons = get_polygons(
            mask,
            MIN_AREA
        )

        for polygon in polygons:

            add_prism(
                model,
                polygon,
                w,
                h,
                carpet_z,
                carpet_z + ornament_z,
                material
            )

            details += 1

    if WOOL:

        add_wool(
            model,
            selected,
            masks,
            w,
            h,
            carpet_z,
            ornament_z,
