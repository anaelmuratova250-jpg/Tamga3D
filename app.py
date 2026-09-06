import io, zipfile, cv2
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
st.write("2D орнамент → тонкая цветная 3D-модель")


# =========================
# IMAGE
# =========================

def load_image(file):
    return np.array(
        Image.open(file).convert("RGB")
    )


def perspective_crop(img, points):
    if len(points) != 4:
        return None

    pts = np.array(points, np.float32)
    center = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    pts = pts[np.argsort(angles)]

    start = np.argmin(
        pts[:, 0] + pts[:, 1]
    )

    pts = np.roll(pts, -start, axis=0)

    p1, p2, p3, p4 = pts

    w = int(max(
        np.linalg.norm(p2 - p1),
        np.linalg.norm(p3 - p4)
    ))

    h = int(max(
        np.linalg.norm(p4 - p1),
        np.linalg.norm(p3 - p2)
    ))

    if w < 10 or h < 10:
        return None

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], np.float32)

    M = cv2.getPerspectiveTransform(
        pts, dst
    )

    return cv2.warpPerspective(
        img, M, (w, h)
    )


# =========================
# COLORS
# =========================

def colors_and_masks(img, n):
    small = cv2.resize(
        img, (120, 120)
    )

    data = small.reshape(
        -1, 3
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        data,
        n,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    q = centers[
        labels.flatten()
    ].reshape(small.shape)

    q = cv2.resize(
        q,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    masks = []

    for color in centers:

        d = np.linalg.norm(
            q.astype(np.float32) -
            color.astype(np.float32),
            axis=2
        )

        mask = np.where(
            d < 5, 255, 0
        ).astype(np.uint8)

        kernel = np.ones(
            (3, 3), np.uint8
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

    return masks, centers


def get_polygons(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = []

    for contour in contours:

        if cv2.contourArea(
            contour
        ) < min_area:
            continue

        eps = (
            0.01 *
            cv2.arcLength(
                contour, True
            )
        )

        poly = cv2.approxPolyDP(
            contour,
            eps,
            True
        ).reshape(-1, 2)

        if len(poly) >= 3:
            result.append(poly)

    return result


# =========================
# TRIANGULATION
# =========================

def inside_triangle(a, b, c, p):

    def s(p1, p2, p3):
        return (
            (p1[0] - p3[0]) *
            (p2[1] - p3[1])
            -
            (p2[0] - p3[0]) *
            (p1[1] - p3[1])
        )

    d1 = s(p, a, b)
    d2 = s(p, b, c)
    d3 = s(p, c, a)

    return not (
        (d1 < 0 or d2 < 0 or d3 < 0)
        and
        (d1 > 0 or d2 > 0 or d3 > 0)
    )


def triangulate(points):
    pts = [
        tuple(map(float, p))
        for p in points
    ]

    if len(pts) == 3:
        return [(0, 1, 2)]

    if len(pts) < 3:
        return []

    area = sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1]
        -
        pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )

    if area < 0:
        pts.reverse()

    ids = list(range(len(pts)))
    result = []

    while len(ids) > 3:

        found = False

        for k in range(len(ids)):

            a = ids[k - 1]
            b = ids[k]
            c = ids[(k + 1) % len(ids)]

            A, B, C = (
                pts[a],
                pts[b],
                pts[c]
            )

            cross = (
                (B[0] - A[0]) *
                (C[1] - A[1])
                -
                (B[1] - A[1]) *
                (C[0] - A[0])
            )

            if cross <= 0:
                continue

            bad = False

            for q in ids:

                if q in (a, b, c):
                    continue

                if inside_triangle(
                    A, B, C, pts[q]
                ):
                    bad = True
                    break

            if bad:
                continue

            result.append((a, b, c))
            ids.pop(k)
            found = True
            break

        if not found:
            break

    if len(ids) == 3:
        result.append(tuple(ids))

    return result


# =========================
# MODEL
# =========================

class Model:

    def __init__(self):
        self.v = []
        self.f = []

    def vertex(self, x, y, z):
        self.v.append(
            (float(x), float(y), float(z))
        )
        return len(self.v)

    def face(self, a, b, c, mat):
        self.f.append(
            (a, b, c, mat)
        )


def add_box(model, thickness):
    z = thickness

    v = [
        model.vertex(-.5, -.5, 0),
        model.vertex(.5, -.5, 0),
        model.vertex(.5, .5, 0),
        model.vertex(-.5, .5, 0),
        model.vertex(-.5, -.5, z),
        model.vertex(.5, -.5, z),
        model.vertex(.5, .5, z),
        model.vertex(-.5, .5, z)
    ]

    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7)
    ]

    for a, b, c in faces:
        model.face(
            v[a], v[b], v[c], 0
        )


def add_prism(
    model,
    polygon,
    size,
    height,
    material,
    base
):

    poly = np.asarray(
        polygon,
        float
    )

    bottom = []
    top = []

    for x, y in poly:

        px = x / size - .5
        py = y / size - .5

        bottom.append(
            model.vertex(
                px, -py, base
            )
        )

        top.append(
            model.vertex(
                px, -py,
                base + height
            )
        )

    for a, b, c in triangulate(poly):

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

    n = len(poly)

    for i in range(n):

        j = (i + 1) % n

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


# =========================
# EXPORT
# =========================

def make_obj(model):

    out = [
        "mtllib Tamga3D_model.mtl",
        "o Tamga3D_Carpet"
    ]

    for x, y, z in model.v:
        out.append(
            f"v {x:.6f} {y:.6f} {z:.6f}"
        )

    current = -1

    for a, b, c, mat in model.f:

        if mat != current:
            out.append(
                f"usemtl material_{mat}"
            )
            current = mat

        out.append(
            f"f {a} {b} {c}"
        )

    return "\n".join(out)


def make_mtl(colors):

    out = []

    for i, color in enumerate(colors):

        r, g, b = (
            np.array(color) / 255
        )

        out += [
            f"newmtl material_{i}",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",
            "Ka 0.05 0.05 0.05",
            "Ks 0.1 0.1 0.1",
            "Ns 20",
            ""
        ]

    return "\n".join(out)


def make_zip(obj, mtl, img):

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        z.writestr(
            "Tamga3D_model.obj",
            obj
        )

        z.writestr(
            "Tamga3D_model.mtl",
            mtl
        )

        png = io.BytesIO()

        Image.fromarray(
            img
        ).save(
            png,
            "PNG"
        )

        z.writestr(
            "Tamga3D_original.png",
            png.getvalue()
        )

    buf.seek(0)
    return buf


# =========================
# SETTINGS
# =========================

st.sidebar.header("⚙️ Настройки")

color_count = st.sidebar.slider(
    "Количество цветов",
    2, 8, 5
)

min_area = st.sidebar.slider(
    "Минимальная площадь",
    20, 1000, 100
)

carpet_mm = st.sidebar.slider(
    "Толщина ковра, мм",
    1.0, 8.0, 3.0, .5
)

ornament_mm = st.sidebar.slider(
    "Высота орнамента, мм",
    .5, 8.0, 3.0, .5
)


# =========================
# UPLOAD
# =========================

uploaded = st.file_uploader(
    "Загрузи фотографию ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info(
        "Сначала загрузи фотографию."
    )
    st.stop()

image = load_image(uploaded)

st.subheader(
    "1. Исходное изображение"
)

st.image(
    image,
    use_container_width=True
)


# =========================
# FOUR POINTS
# =========================

st.subheader(
    "2. Выбери 4 угла области"
)

display_width = min(
    700,
    image.shape[1]
)

scale = (
    display_width /
    image.shape[1]
)

display = cv2.resize(
    image,
    (
        display_width,
        int(image.shape[0] * scale)
    )
)

if "points" not in st.session_state:
    st.session_state.points = []

coords = streamlit_image_coordinates(
    Image.fromarray(display),
    key="select_area"
)

if coords:

    point = (
        int(coords["x"] / scale),
        int(coords["y"] / scale)
    )

    if (
        len(st.session_state.points) < 4
        and point not in
        st.session_state.points
    ):

        st.session_state.points.append(
            point
        )

        st.rerun()


st.write(
    f"Выбрано: "
    f"**{len(st.session_state.points)}/4**"
)

for i, p in enumerate(
    st.session_state.points
):
    st.write(
        f"🔸 {i + 1}: "
        f"X={p[0]}, Y={p[1]}"
    )


if st.button("🔄 Сбросить точки"):

    st.session_state.points = []

    for key in [
        "selected",
        "model",
        "model_colors"
    ]:
        st.session_state.pop(
            key,
            None
        )

    st.rerun()


# =========================
# SELECTED AREA
# =========================

if len(
    st.session_state.points
) == 4:

    selected = perspective_crop(
        image,
        st.session_state.points
    )

    if selected is not None:

        st.session_state.selected = selected

        st.success(
            "✅ 4 точки выбраны!"
        )

        st.image(
            selected,
            caption="Выбранная область",
            use_container_width=True
        )


if "selected" not in st.session_state:

    st.warning(
        "Поставь 4 точки по углам области."
    )

    st.stop()

selected = st.session_state.selected


# =========================
# COLORS
# =========================

st.subheader(
    "3. Распознавание цветов"
)

masks, colors = colors_and_masks(
    selected,
    color_count
)

preview = np.zeros_like(
    selected
)

for mask, color in zip(
    masks, colors
):
    preview[mask > 0] = color

st.image(
    preview,
    caption="Распознанные цвета",
    use_container_width=True
)


# =========================
# CREATE 3D
# =========================

if st.button(
    "🧶 Создать 3D-модель",
    type="primary"
):

    h, w = selected.shape[:2]

    model = Model()

    base = carpet_mm / 1000
    height = ornament_mm / 1000

    add_box(
        model,
        base
    )

    count = 0

    for material, mask in enumerate(
        masks
    ):

        polygons = get_polygons(
            mask,
            min_area
        )

        for polygon in polygons:

            add_prism(
                model,
                polygon,
                max(w, h),
                height,
                material,
                base
            )

            count += 1

    st.session_state.model = model
    st.session_state.model_colors = colors

    st.success(
        f"Готово! Деталей: {count}"
    )


# =========================
# 3D PREVIEW
# =========================

if "model" in st.session_state:

    model = st.session_state.model
    colors = st.session_state.model_colors

    st.subheader(
        "4. 3D-предпросмотр"
    )

    verts = np.array(
        model.v
    )

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    ii, jj, kk = [], [], []
    face_colors = []

    for a, b, c, material in model.f:

        ii.append(a - 1)
        jj.append(b - 1)
        kk.append(c - 1)

        col = colors[
            material % len(colors)
        ]

        face_colors.append(
            "rgb(%d,%d,%d)" %
            tuple(col)
        )

    fig = go.Figure(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=ii,
            j=jj,
            k=kk,
            facecolor=face_colors,
            flatshading=True
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================
# EXPORT
# =========================

    st.subheader(
        "5. Экспорт для 3ds Max"
    )

    obj = make_obj(model)
    mtl = make_mtl(colors)

    archive = make_zip(
        obj,
        mtl,
        selected
    )

    st.download_button(
        "⬇️ Скачать модель для 3ds Max",
        archive,
        "Tamga3D_model.zip",
        "application/zip"
    )

    st.caption(
        "ZIP содержит OBJ, MTL и PNG."
)
