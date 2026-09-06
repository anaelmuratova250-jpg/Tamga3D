import io, zipfile, cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Tamga3D", page_icon="🧶", layout="wide")
st.title("🧶 Tamga3D")
st.write("Фото ковра → тонкая цветная 3D-модель")


# ═══════════════════════════════════════
# IMAGE
# ═══════════════════════════════════════

def load_image(file):
    return np.array(Image.open(file).convert("RGB"))


def prepare_image(img, size=180):
    h, w = img.shape[:2]
    scale = min(size / w, size / h, 1)

    nw = max(20, int(w * scale))
    nh = max(20, int(h * scale))

    return cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


# ═══════════════════════════════════════
# COLORS
# ═══════════════════════════════════════

def reduce_colors(img, count):
    small = cv2.resize(img, (100, 100))
    data = np.float32(small.reshape(-1, 3))

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        25,
        1
    )

    _, labels, centers = cv2.kmeans(
        data,
        count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    labels = labels.reshape(100, 100)

    result = centers[labels]

    result = cv2.resize(
        result,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return result, centers


# ═══════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════

def make_grid(img, step=3):
    h, w = img.shape[:2]

    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)

    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)

    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)

    vertices = []
    faces = []
    materials = []

    sx = max(w, h)

    for y in ys:
        for x in xs:
            px = x / sx - w / sx / 2
            py = -(y / sx - h / sx / 2)

            vertices.append(
                [px, py, 0]
            )

    cols = len(xs)

    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):

            a = j * cols + i
            b = a + 1
            c = a + cols
            d = c + 1

            faces.append([a, b, d])
            faces.append([a, d, c])

            color = img[
                (ys[j] + ys[j + 1]) // 2,
                (xs[i] + xs[i + 1]) // 2
            ]

            materials.append(tuple(color))
            materials.append(tuple(color))

    return (
        np.array(vertices, dtype=float),
        np.array(faces, dtype=int),
        materials
    )


# ═══════════════════════════════════════
# THICKNESS
# ═══════════════════════════════════════

def add_thickness(vertices, faces, thickness):
    top = vertices.copy()
    bottom = vertices.copy()

    top[:, 2] = thickness

    verts = np.vstack([bottom, top])

    n = len(vertices)

    all_faces = []
    mats = []

    # bottom
    for f in faces:
        all_faces.append(
            [f[2], f[1], f[0]]
        )

    # top
    for f in faces:
        all_faces.append(
            [f[0] + n, f[1] + n, f[2] + n]
        )

    # sides
    edges = {}

    for f in faces:
        for i in range(3):
            a = int(f[i])
            b = int(f[(i + 1) % 3])

            key = tuple(sorted((a, b)))

            if key in edges:
                edges[key] = False
            else:
                edges[key] = True

    for (a, b), outside in edges.items():
        if not outside:
            continue

        all_faces.append(
            [a, b, b + n]
        )

        all_faces.append(
            [a, b + n, a + n]
        )

    return verts, np.array(all_faces, dtype=int)


# ═══════════════════════════════════════
# OBJ / MTL
# ═══════════════════════════════════════

def color_name(c):
    return "c_%d_%d_%d" % tuple(c)


def make_obj(vertices, faces, colors):
    lines = [
        "mtllib carpet.mtl"
    ]

    for v in vertices:
        lines.append(
            "v %.6f %.6f %.6f" %
            tuple(v)
        )

    last = None

    for face, color in zip(faces, colors):

        name = color_name(color)

        if name != last:
            lines.append(
                "usemtl " + name
            )
            last = name

        a, b, c = face + 1

        lines.append(
            f"f {a} {b} {c}"
        )

    return "\n".join(lines)


def make_mtl(colors):
    unique = {}

    for c in colors:
        unique[color_name(c)] = c

    lines = []

    for name, c in unique.items():

        r, g, b = np.array(c) / 255

        lines += [
            f"newmtl {name}",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",
            "Ka 0.05 0.05 0.05",
            "Ks 0.05 0.05 0.05",
            "Ns 8",
            ""
        ]

    return "\n".join(lines)


def export_zip(vertices, faces, colors):
    obj = make_obj(
        vertices,
        faces,
        colors
    ).encode()

    mtl = make_mtl(colors).encode()

    data = io.BytesIO()

    with zipfile.ZipFile(
        data,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        z.writestr(
            "Tamga3D_carpet.obj",
            obj
        )

        z.writestr(
            "carpet.mtl",
            mtl
        )

    data.seek(0)

    return data


# ═══════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════

st.sidebar.header("⚙️ Настройки")

colors_count = st.sidebar.slider(
    "Цветов",
    4,
    12,
    7
)

grid_step = st.sidebar.slider(
    "Детализация 3D",
    2,
    6,
    3
)

thickness_mm = st.sidebar.slider(
    "Толщина ковра, мм",
    1.0,
    6.0,
    3.0,
    0.5
)


# ═══════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════

file = st.file_uploader(
    "🖼️ Загрузите фото ковра",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

if file is None:
    st.info(
        "Загрузите фотографию — "
        "3D-модель создастся автоматически."
    )
    st.stop()


# ═══════════════════════════════════════
# PROCESS
# ═══════════════════════════════════════

image = load_image(file)

st.subheader("Исходное фото")
st.image(
    image,
    use_container_width=True
)


with st.spinner("⚙️ Подготавливаю изображение..."):

    carpet = prepare_image(
        image,
        180
    )


with st.spinner("🎨 Определяю цвета..."):

    colored, palette = reduce_colors(
        carpet,
        colors_count
    )


st.subheader("🎨 Цветовая карта ковра")

st.image(
    colored,
    use_container_width=True
)


# ═══════════════════════════════════════
# MESH
# ═══════════════════════════════════════

with st.spinner("🧶 Создаю лёгкую 3D-сетку..."):

    vertices, faces, face_colors = make_grid(
        colored,
        grid_step
    )

    thickness = thickness_mm / 1000

    vertices, faces = add_thickness(
        vertices,
        faces,
        thickness
    )


# Для нижней и боковой части
# используем тёмный цвет.
side_color = (45, 45, 45)

top_count = len(
    face_colors
)

all_colors = []

for i in range(len(faces)):

    if i < top_count * 2:
        color = face_colors[
            i % top_count
        ]
    else:
        color = side_color

    all_colors.append(
        tuple(map(int, color))
    )


# ═══════════════════════════════════════
# PREVIEW COLORS
# ═══════════════════════════════════════

# Plotly не всегда хорошо показывает
# множество материалов Mesh3d,
# поэтому создаём отдельные лёгкие
# поверхности по цветам.

fig = go.Figure()

unique_colors = {}

for i, c in enumerate(face_colors):

    unique_colors.setdefault(
        tuple(map(int, c)),
        []
    ).append(i)


top_vertices = vertices[
    len(vertices) // 2:
]

for color, ids in unique_colors.items():

    ids = np.array(ids)

    fs = faces[
        ids + top_count
    ]

    if len(fs) == 0:
        continue

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=fs[:, 0],
            j=fs[:, 1],
            k=fs[:, 2],
            color=(
                "rgb(%d,%d,%d)" %
                color
            ),
            flatshading=True,
            hoverinfo="skip",
            showscale=False
        )
    )


fig.update_layout(
    height=600,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),
    scene=dict(
        aspectmode="data",
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        zaxis=dict(
            visible=False
        )
    )
)


st.subheader("🧶 3D-модель")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True
    }
)


# ═══════════════════════════════════════
# INFO
# ═══════════════════════════════════════

st.success(
    "✅ Модель готова"
)

st.write(
    f"Вершин: {len(vertices):,}"
)

st.write(
    f"Полигонов: {len(faces):,}"
)

st.write(
    "Модель сделана тонкой, "
    "как готовый ковёр."
)


# ═══════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════

zip_file = export_zip(
    vertices,
    faces,
    all_colors
)


st.download_button(
    "⬇️ Скачать модель для 3ds Max",
    data=zip_file,
    file_name="Tamga3D_carpet.zip",
    mime="application/zip"
)

st.caption(
    "Экспорт содержит OBJ + MTL. "
    "Цвета ковра записываются в материалы."
    )
