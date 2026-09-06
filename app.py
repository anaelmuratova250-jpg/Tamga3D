import io
import zipfile
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="Tamga3D",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Tamga3D")
st.write("2D орнамент → простой цветной 3D-модель")


# ---------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------

def image_to_array(uploaded):
    image = Image.open(uploaded).convert("RGB")
    return np.array(image)


def crop_rectangle(image, points):
    x1, y1 = points[0]
    x2, y2 = points[1]

    left = int(min(x1, x2))
    right = int(max(x1, x2))
    top = int(min(y1, y2))
    bottom = int(max(y1, y2))

    if right <= left or bottom <= top:
        return None

    return image[top:bottom, left:right]


def reduce_colors(image, colors):
    small = cv2.resize(image, (120, 120))

    data = small.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        data,
        colors,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(small.shape)

    result = cv2.resize(
        result,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return result, labels.reshape((120, 120)), centers


def get_color_masks(image, colors):
    quantized, _, centers = reduce_colors(image, colors)

    masks = []

    for color in centers:
        diff = np.linalg.norm(
            quantized.astype(np.float32) -
            color.astype(np.float32),
            axis=2
        )

        mask = np.where(diff < 5, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)

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

    return masks, centers


def polygon_from_mask(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        epsilon = 0.01 * cv2.arcLength(
            contour,
            True
        )

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(polygon) >= 3:
            polygons.append(
                polygon.reshape(-1, 2)
            )

    return polygons


def point_inside_triangle(a, b, c, p):
    def sign(p1, p2, p3):
        return (
            (p1[0] - p3[0]) * (p2[1] - p3[1])
            -
            (p2[0] - p3[0]) * (p1[1] - p3[1])
        )

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    negative = d1 < 0 or d2 < 0 or d3 < 0
    positive = d1 > 0 or d2 > 0 or d3 > 0

    return not (negative and positive)


def triangulate_polygon(points):
    points = [tuple(map(float, p)) for p in points]

    if len(points) < 3:
        return []

    if len(points) == 3:
        return [(0, 1, 2)]

    area = 0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1

    if area < 0:
        points.reverse()

    remaining = list(range(len(points)))
    triangles = []

    guard = 0

    while len(remaining) > 3 and guard < 10000:
        guard += 1
        ear_found = False

        for i in range(len(remaining)):
            prev_i = remaining[i - 1]
            curr_i = remaining[i]
            next_i = remaining[(i + 1) % len(remaining)]

            a = points[prev_i]
            b = points[curr_i]
            c = points[next_i]

            cross = (
                (b[0] - a[0]) * (c[1] - a[1])
                -
                (b[1] - a[1]) * (c[0] - a[0])
            )

            if cross <= 0:
                continue

            contains = False

            for other in remaining:
                if other in (prev_i, curr_i, next_i):
                    continue

                if point_inside_triangle(
                    a,
                    b,
                    c,
                    points[other]
                ):
                    contains = True
                    break

            if contains:
                continue

            triangles.append(
                (prev_i, curr_i, next_i)
            )

            remaining.pop(i)
            ear_found = True
            break

        if not ear_found:
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


# ---------------------------------------------------------
# 3D MODEL
# ---------------------------------------------------------

class Model:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.materials = []

    def vertex(self, x, y, z):
        self.vertices.append(
            (float(x), float(y), float(z))
        )
        return len(self.vertices)

    def face(self, a, b, c, material):
        self.faces.append(
            (a, b, c, material)
        )


def add_box(model, x1, y1, x2, y2, z1, z2, material):
    v1 = model.vertex(x1, y1, z1)
    v2 = model.vertex(x2, y1, z1)
    v3 = model.vertex(x2, y2, z1)
    v4 = model.vertex(x1, y2, z1)

    v5 = model.vertex(x1, y1, z2)
    v6 = model.vertex(x2, y1, z2)
    v7 = model.vertex(x2, y2, z2)
    v8 = model.vertex(x1, y2, z2)

    model.face(v1, v3, v2, material)
    model.face(v1, v4, v3, material)

    model.face(v5, v6, v7, material)
    model.face(v5, v7, v8, material)

    model.face(v1, v2, v6, material)
    model.face(v1, v6, v5, material)

    model.face(v2, v3, v7, material)
    model.face(v2, v7, v6, material)

    model.face(v3, v4, v8, material)
    model.face(v3, v8, v7, material)

    model.face(v4, v1, v5, material)
    model.face(v4, v5, v8, material)


def add_polygon_prism(
    model,
    polygon,
    width,
    height,
    material,
    base_z
):
    polygon = np.asarray(polygon, dtype=float)

    if len(polygon) < 3:
        return

    center = polygon.mean(axis=0)

    top_indices = []
    bottom_indices = []

    for x, y in polygon:
        px = (x / width) - 0.5
        py = (y / width) - 0.5

        bottom_indices.append(
            model.vertex(
                px,
                -py,
                base_z
            )
        )

        top_indices.append(
            model.vertex(
                px,
                -py,
                base_z + height
            )
        )

    triangles = triangulate_polygon(polygon)

    for a, b, c in triangles:
        model.face(
            top_indices[a],
            top_indices[b],
            top_indices[c],
            material
        )

        model.face(
            bottom_indices[c],
            bottom_indices[b],
            bottom_indices[a],
            material
        )

    count = len(polygon)

    for i in range(count):
        j = (i + 1) % count

        model.face(
            bottom_indices[i],
            bottom_indices[j],
            top_indices[j],
            material
        )

        model.face(
            bottom_indices[i],
            top_indices[j],
            top_indices[i],
            material
        )


# ---------------------------------------------------------
# OBJ + MTL
# ---------------------------------------------------------

def make_obj(model):
    lines = []

    lines.append("mtllib Tamga3D_model.mtl")
    lines.append("o Tamga3D_Carpet")

    for x, y, z in model.vertices:
        lines.append(
            f"v {x:.6f} {y:.6f} {z:.6f}"
        )

    current_material = None

    for a, b, c, material in model.faces:
        if material != current_material:
            lines.append(
                f"usemtl material_{material}"
            )
            current_material = material

        lines.append(
            f"f {a} {b} {c}"
        )

    return "\n".join(lines)


def make_mtl(colors):
    lines = []

    for i, color in enumerate(colors):
        r = int(color[0]) / 255
        g = int(color[1]) / 255
        b = int(color[2]) / 255

        lines.append(
            f"newmtl material_{i}"
        )

        lines.append(
            f"Kd {r:.4f} {g:.4f} {b:.4f}"
        )

        lines.append("Ka 0.05 0.05 0.05")
        lines.append("Ks 0.1 0.1 0.1")
        lines.append("Ns 20")
        lines.append("")

    return "\n".join(lines)


def make_zip(obj_text, mtl_text, image):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        z.writestr(
            "Tamga3D_model.obj",
            obj_text
        )

        z.writestr(
            "Tamga3D_model.mtl",
            mtl_text
        )

        png_buffer = io.BytesIO()

        Image.fromarray(image).save(
            png_buffer,
            format="PNG"
        )

        z.writestr(
            "Tamga3D_original.png",
            png_buffer.getvalue()
        )

    buffer.seek(0)

    return buffer


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Настройки")

color_count = st.sidebar.slider(
    "Количество цветов",
    2,
    8,
    5
)

min_area = st.sidebar.slider(
    "Минимальная площадь орнамента",
    20,
    1000,
    100
)

carpet_thickness_mm = st.sidebar.slider(
    "Толщина ковра, мм",
    1.0,
    8.0,
    3.0,
    0.5
)

ornament_height_mm = st.sidebar.slider(
    "Высота орнамента, мм",
    0.5,
    8.0,
    3.0,
    0.5
)

st.sidebar.info(
    "Выбери область ковра, затем нажми кнопку создания 3D."
)


# ---------------------------------------------------------
# ЗАГРУЗКА
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "Загрузи фотографию ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Сначала загрузи фотографию ковра.")
    st.stop()

image = image_to_array(uploaded)

st.subheader("1. Исходное изображение")

st.image(
    image,
    use_container_width=True
)


# ---------------------------------------------------------
# ВЫБОР ОБЛАСТИ
# ---------------------------------------------------------

st.subheader("2. Выбери часть ковра")

display_width = min(700, image.shape[1])

scale = display_width / image.shape[1]

display_height = int(
    image.shape[0] * scale
)

display_image = cv2.resize(
    image,
    (display_width, display_height)
)

coords = streamlit_image_coordinates(
    Image.fromarray(display_image),
    key="select_area"
)

if coords:
    x = coords["x"]
    y = coords["y"]

    st.write(
        f"Последняя точка: X={x}, Y={y}"
    )

if "points" not in st.session_state:
    st.session_state.points = []

if coords:
    point = (
        int(coords["x"] / scale),
        int(coords["y"] / scale)
    )

    if not st.session_state.points:
        st.session_state.points.append(point)

    elif st.session_state.points[-1] != point:
        st.session_state.points.append(point)

    if len(st.session_state.points) > 2:
        st.session_state.points = (
            st.session_state.points[-2:]
        )

if len(st.session_state.points) == 2:
    p1, p2 = st.session_state.points

    selected = crop_rectangle(
        image,
        [p1, p2]
    )

    if selected is not None:
        st.session_state.selected = selected

        st.success("Область выбрана.")

        st.image(
            selected,
            caption="Выбранная область",
            use_container_width=True
        )

        if st.button(
            "🔄 Выбрать область заново"
        ):
            st.session_state.points = []
            st.session_state.pop(
                "selected",
                None
            )
            st.rerun()


# ---------------------------------------------------------
# СОЗДАНИЕ 3D
# ---------------------------------------------------------

if "selected" not in st.session_state:
    st.warning(
        "Выбери область двумя точками."
    )
    st.stop()

selected = st.session_state.selected

st.subheader("3. Распознавание цветов")

masks, colors = get_color_masks(
    selected,
    color_count
)

preview = np.zeros_like(selected)

for mask, color in zip(masks, colors):
    preview[mask > 0] = color

st.image(
    preview,
    caption="Распознанные цвета",
    use_container_width=True
)


# ---------------------------------------------------------
# КНОПКА
# ---------------------------------------------------------

if st.button(
    "🧶 Создать 3D-модель",
    type="primary"
):

    h, w = selected.shape[:2]

    model = Model()

    carpet_z = carpet_thickness_mm / 1000
    ornament_z = ornament_height_mm / 1000

    # Основа ковра
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

    polygons_all = []

    for color_index, mask in enumerate(masks):

        polygons = polygon_from_mask(
            mask,
            min_area
        )

        for polygon in polygons:

            # Не превращаем слишком маленькие куски
            # в отдельные 3D детали
            if cv2.contourArea(
                polygon.astype(np.float32)
            ) < min_area:
                continue

            polygons_all.append(
                (
                    polygon,
                    color_index
                )
            )

    for polygon, color_index in polygons_all:

        add_polygon_prism(
            model,
            polygon,
            max(w, h),
            ornament_z,
            color_index,
            carpet_z
        )

    model.materials = colors

    st.session_state.model = model
    st.session_state.model_colors = colors

    st.success(
        f"Готово! Создано деталей орнамента: "
        f"{len(polygons_all)}"
    )


# ---------------------------------------------------------
# ПРЕДПРОСМОТР
# ---------------------------------------------------------

if "model" in st.session_state:

    model = st.session_state.model
    colors = st.session_state.model_colors

    st.subheader("4. 3D-предпросмотр")

    vertices = np.array(
        model.vertices
    )

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    i = []
    j = []
    k = []
    face_colors = []

    for a, b, c, material in model.faces:

        i.append(a - 1)
        j.append(b - 1)
        k.append(c - 1)

        color = colors[
            material % len(colors)
        ]

        face_colors.append(
            "rgb(%d,%d,%d)" %
            (
                color[0],
                color[1],
                color[2]
            )
        )

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                facecolor=face_colors,
                flatshading=True,
                opacity=1.0
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------------------------------------------------
# ЭКСПОРТ
# ---------------------------------------------------------

    st.subheader("5. Экспорт для 3ds Max")

    obj_text = make_obj(model)
    mtl_text = make_mtl(colors)

    archive = make_zip(
        obj_text,
        mtl_text,
        selected
    )

    st.download_button(
        label="⬇️ Скачать 3D-модель для 3ds Max",
        data=archive,
        file_name="Tamga3D_model.zip",
        mime="application/zip"
    )

    st.caption(
        "ZIP содержит OBJ, MTL и исходное изображение."
    )
