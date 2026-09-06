import io, zipfile, cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Tamga3D", page_icon="🧶", layout="wide")
st.title("🧶 Tamga3D")
st.write("Фото ковра → готовая тонкая 3D-модель")

# ---------- IMAGE ----------
def load_image(file):
    return np.array(Image.open(file).convert("RGB"))


def prepare_carpet(img):
    """Автоматически находит основной прямоугольник ковра."""
    h, w = img.shape[:2]
    scale = min(800 / w, 800 / h, 1)

    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)

        area = cw * ch
        if area > small.shape[0] * small.shape[1] * 0.2:
            x = int(x / scale)
            y = int(y / scale)
            cw = int(cw / scale)
            ch = int(ch / scale)

            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            cw = min(w - x, cw + pad * 2)
            ch = min(h - y, ch + pad * 2)

            return img[y:y + ch, x:x + cw]

    return img


# ---------- COLORS ----------
def reduce_colors(img, n):
    h, w = img.shape[:2]
    small = cv2.resize(img, (100, 100))
    data = np.float32(small.reshape(-1, 3))

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0
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
    labels = labels.reshape(100, 100)

    result = centers[labels]
    result = cv2.resize(
        result,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return result, centers


def make_masks(img, centers):
    masks = []

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    for color in centers:
        c = np.uint8([[color]])
        clab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB)[0, 0]

        dist = np.sqrt(
            np.sum((lab.astype(float) - clab.astype(float)) ** 2, axis=2)
        )

        mask = np.uint8(dist < 35) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        masks.append(mask)

    return masks


# ---------- POLYGONS ----------
def get_polygons(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area:
            continue

        eps = 0.015 * cv2.arcLength(c, True)
        p = cv2.approxPolyDP(c, eps, True)
        p = p.reshape(-1, 2)

        if len(p) >= 3:
            polygons.append(p)

    return polygons


def triangle_area(a, b, c):
    return abs(
        (b[0] - a[0]) * (c[1] - a[1]) -
        (b[1] - a[1]) * (c[0] - a[0])
    ) / 2


def point_inside(p, a, b, c):
    A = triangle_area(a, b, c)
    A1 = triangle_area(p, b, c)
    A2 = triangle_area(a, p, c)
    A3 = triangle_area(a, b, p)

    return abs(A - A1 - A2 - A3) < 0.5


def triangulate(poly):
    if len(poly) == 3:
        return [(0, 1, 2)]

    points = [tuple(p) for p in poly]
    result = []

    if cv2.contourArea(poly.astype(np.float32), oriented=True) < 0:
        points.reverse()

    ids = list(range(len(points)))

    guard = 0

    while len(ids) > 3 and guard < 1000:
        guard += 1
        found = False

        for i in range(len(ids)):
            a = ids[i - 1]
            b = ids[i]
            c = ids[(i + 1) % len(ids)]

            A, B, C = points[a], points[b], points[c]

            cross = (
                (B[0] - A[0]) * (C[1] - A[1]) -
                (B[1] - A[1]) * (C[0] - A[0])
            )

            if cross <= 0:
                continue

            good = True

            for j in ids:
                if j in (a, b, c):
                    continue

                if point_inside(points[j], A, B, C):
                    good = False
                    break

            if good:
                result.append((a, b, c))
                ids.pop(i)
                found = True
                break

        if not found:
            break

    if len(ids) == 3:
        result.append(tuple(ids))

    return result


# ---------- MODEL ----------
class Model:
    def __init__(self):
        self.v = []
        self.f = []
        self.m = []

    def vertex(self, x, y, z):
        self.v.append((x, y, z))
        return len(self.v)

    def face(self, a, b, c, material):
        self.f.append((a, b, c))
        self.m.append(material)


def add_box(model, w, h, z):
    a = model.vertex(-w/2, -h/2, 0)
    b = model.vertex(w/2, -h/2, 0)
    c = model.vertex(w/2, h/2, 0)
    d = model.vertex(-w/2, h/2, 0)

    e = model.vertex(-w/2, -h/2, z)
    f = model.vertex(w/2, -h/2, z)
    g = model.vertex(w/2, h/2, z)
    h2 = model.vertex(-w/2, h/2, z)

    model.face(a, b, c, 0)
    model.face(a, c, d, 0)

    model.face(e, g, f, 0)
    model.face(e, h2, g, 0)

    model.face(a, e, f, 0)
    model.face(a, f, b, 0)

    model.face(b, f, g, 0)
    model.face(b, g, c, 0)

    model.face(c, g, h2, 0)
    model.face(c, h2, d, 0)

    model.face(d, h2, e, 0)
    model.face(d, e, a, 0)


def add_polygon(model, poly, color_id, w, h, depth):
    scale = max(w, h)

    top = []

    for x, y in poly:
        px = x / scale - w / scale / 2
        py = -(y / scale - h / scale / 2)
        top.append(
            model.vertex(px, py, depth)
        )

    bottom = []

    for x, y in poly:
        px = x / scale - w / scale / 2
        py = -(y / scale - h / scale / 2)
        bottom.append(
            model.vertex(px, py, 0)
        )

    tris = triangulate(poly)

    for a, b, c in tris:
        model.face(
            top[a],
            top[b],
            top[c],
            color_id + 1
        )

        model.face(
            bottom[c],
            bottom[b],
            bottom[a],
            color_id + 1
        )

    for i in range(len(poly)):
        j = (i + 1) % len(poly)

        model.face(
            bottom[i],
            bottom[j],
            top[j],
            color_id + 1
        )

        model.face(
            bottom[i],
            top[j],
            top[i],
            color_id + 1
        )


# ---------- EXPORT ----------
def make_obj(model):
    out = ["mtllib carpet.mtl"]

    for x, y, z in model.v:
        out.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    current = -1

    for face, material in zip(model.f, model.m):
        if material != current:
            out.append(f"usemtl color_{material}")
            current = material

        a, b, c = face
        out.append(f"f {a} {b} {c}")

    return "\n".join(out)


def make_mtl(colors):
    out = []

    for i, color in enumerate(colors):
        r, g, b = np.array(color) / 255
        out.append(f"newmtl color_{i}")
        out.append(f"Kd {r:.4f} {g:.4f} {b:.4f}")
        out.append("Ka 0.1 0.1 0.1")
        out.append("Ks 0.05 0.05 0.05")
        out.append("Ns 10")
        out.append("")

    return "\n".join(out)


def make_zip(model, colors):
    obj = make_obj(model).encode()
    mtl = make_mtl(colors).encode()

    data = io.BytesIO()

    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Tamga3D_carpet.obj", obj)
        z.writestr("carpet.mtl", mtl)

    data.seek(0)
    return data


# ---------- SETTINGS ----------
st.sidebar.header("⚙️ Настройки")

color_count = st.sidebar.slider(
    "Количество цветов",
    4, 16, 8
)

min_area = st.sidebar.slider(
    "Минимальный орнамент",
    10, 500, 40
)

carpet_thickness = st.sidebar.slider(
    "Толщина ковра, мм",
    1.0, 8.0, 3.0, 0.5
)

relief_height = st.sidebar.slider(
    "Высота орнамента, мм",
    0.2, 5.0, 1.0, 0.1
)


# ---------- APP ----------
file = st.file_uploader(
    "🖼️ Загрузите фотографию ковра",
    type=["jpg", "jpeg", "png", "webp"]
)

if not file:
    st.info("Загрузите изображение — модель создастся автоматически.")
    st.stop()

image = load_image(file)

st.subheader("Исходное изображение")
st.image(image, use_container_width=True)

with st.spinner("🔍 Распознаю ковёр..."):
    carpet = prepare_carpet(image)

st.subheader("Область ковра")
st.image(carpet, use_container_width=True)

with st.spinner("🎨 Распознаю цвета..."):
    quantized, colors = reduce_colors(
        carpet,
        color_count
    )

st.subheader("🎨 Распознанные цвета")
st.image(
    quantized,
    use_container_width=True
)

masks = make_masks(
    quantized,
    colors
)

h, w = carpet.shape[:2]

# миллиметры → условные метры
base_z = carpet_thickness / 1000
ornament_z = (
    carpet_thickness + relief_height
) / 1000

model = Model()

# тонкая основа
add_box(
    model,
    w / max(w, h),
    h / max(w, h),
    base_z
)

# орнамент
for color_id, mask in enumerate(masks):
    polygons = get_polygons(
        mask,
        min_area
    )

    for poly in polygons:
        if cv2.contourArea(poly) < min_area:
            continue

        add_polygon(
            model,
            poly,
            color_id,
            w,
            h,
            ornament_z
        )

st.success(
    f"✅ 3D-модель готова: "
    f"{len(model.v)} вершин, "
    f"{len(model.f)} полигонов"
)

# ---------- PREVIEW ----------
verts = np.array(model.v)
faces = np.array(model.f)

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0] - 1,
            j=faces[:, 1] - 1,
            k=faces[:, 2] - 1,
            intensity=verts[:, 2],
            colorscale="Viridis",
            showscale=False,
            flatshading=False
        )
    ]
)

fig.update_layout(
    height=650,
    scene=dict(
        aspectmode="data",
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

st.subheader("🧶 Готовая 3D-модель")
st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------- EXPORT ----------
zip_file = make_zip(
    model,
    colors
)

st.download_button(
    "⬇️ Скачать 3D-модель для 3ds Max",
    data=zip_file,
    file_name="Tamga3D_carpet.zip",
    mime="application/zip"
)

st.caption(
    "Модель тонкая: основа + небольшой рельеф орнамента. "
    "Цвета сохраняются через материалы OBJ/MTL."
    )
