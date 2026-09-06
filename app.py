import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
import tempfile
import os
import zipfile


# ============================================================
# НАСТРОЙКИ
# ============================================================

st.set_page_config(
    page_title="Tamga3D",
    layout="wide"
)

st.title("🧿 Tamga3D")
st.write("2D орнамент → цветная тонкая 3D модель")


# ============================================================
# НАСТРОЙКИ 3D
# ============================================================

# 🔵 ФИЗИЧЕСКАЯ ТОЛЩИНА КОВРА
CARPET_THICKNESS = 0.006

# 🟢 МАКСИМАЛЬНАЯ ВЫСОТА РЕЛЬЕФА
RELIEF_HEIGHT = 0.003

# 🟡 КОЛИЧЕСТВО ВЕРШИН
# Больше = меньше размытия, но тяжелее модель
GRID_SIZE = 160


# ============================================================
# SESSION STATE
# ============================================================

if "points" not in st.session_state:
    st.session_state.points = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# ============================================================
# КНОПКА СБРОСА
# ============================================================

if st.button("🔄 Начать выбор заново"):

    st.session_state.points = []
    st.session_state.last_click = None

    st.rerun()


# ============================================================
# ЗАГРУЗКА
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Загрузи фотографию ковра",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:

    st.info("Сначала загрузи фотографию ковра.")

    st.stop()


# ============================================================
# ЧТЕНИЕ ИЗОБРАЖЕНИЯ
# ============================================================

file_bytes = np.asarray(
    bytearray(uploaded_file.read()),
    dtype=np.uint8
)

image_bgr = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_COLOR
)

if image_bgr is None:

    st.error("Не удалось открыть изображение.")

    st.stop()


image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# ВЫБОР 4 ТОЧЕК
# ============================================================

st.subheader("1️⃣ Выбери область орнамента")

st.write(
    "Нажми 4 раза: "
    "левый верх → правый верх → правый низ → левый низ."
)

st.write(
    f"Выбрано точек: {len(st.session_state.points)} / 4"
)


clicked = streamlit_image_coordinates(
    image_rgb,
    key="carpet_image"
)


# ============================================================
# СОХРАНЯЕМ КЛИК
# ============================================================

if clicked is not None:

    new_point = (
        int(clicked["x"]),
        int(clicked["y"])
    )

    last_point = st.session_state.last_click

    if new_point != last_point:

        if len(st.session_state.points) < 4:

            st.session_state.points.append(
                new_point
            )

            st.session_state.last_click = new_point

            st.rerun()


# ============================================================
# ПОКАЗЫВАЕМ ТОЧКИ
# ============================================================

if len(st.session_state.points) > 0:

    st.write("Точки:")

    for i, point in enumerate(
        st.session_state.points
    ):

        st.write(
            f"{i + 1}. X={point[0]}, Y={point[1]}"
        )


# ============================================================
# ЖДЁМ 4 ТОЧКИ
# ============================================================

if len(st.session_state.points) < 4:

    st.info(
        "Поставь ещё "
        + str(4 - len(st.session_state.points))
        + " точки."
    )

    st.stop()


# ============================================================
# КООРДИНАТЫ
# ============================================================

p1 = st.session_state.points[0]
p2 = st.session_state.points[1]
p3 = st.session_state.points[2]
p4 = st.session_state.points[3]


# Берём прямоугольник между точками

x_min = min(
    p1[0],
    p2[0],
    p3[0],
    p4[0]
)

x_max = max(
    p1[0],
    p2[0],
    p3[0],
    p4[0]
)

y_min = min(
    p1[1],
    p2[1],
    p3[1],
    p4[1]
)

y_max = max(
    p1[1],
    p2[1],
    p3[1],
    p4[1]
)


# Проверяем границы

x_min = max(
    0,
    min(x_min, image_rgb.shape[1] - 1)
)

x_max = max(
    1,
    min(x_max, image_rgb.shape[1])
)

y_min = max(
    0,
    min(y_min, image_rgb.shape[0] - 1)
)

y_max = max(
    1,
    min(y_max, image_rgb.shape[0])
)


# ============================================================
# ОБРЕЗКА
# ============================================================

crop = image_rgb[
    y_min:y_max,
    x_min:x_max
]


if crop.size == 0:

    st.error(
        "Не удалось выделить область."
    )

    st.stop()


# ============================================================
# ПРЕДПРОСМОТР
# ============================================================

st.subheader("2️⃣ Выбранный орнамент")

st.image(
    crop,
    use_container_width=True
)


# ============================================================
# СОХРАНЯЕМ ОРИГИНАЛЬНЫЙ ЦВЕТ
# ============================================================

original_crop = crop.copy()


# ============================================================
# ПОДГОТОВКА ТЕКСТУРЫ
# ============================================================

texture = cv2.resize(
    original_crop,
    (
        GRID_SIZE,
        GRID_SIZE
    ),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# КАРТА ВЫСОТЫ
# ============================================================

gray = cv2.cvtColor(
    original_crop,
    cv2.COLOR_RGB2GRAY
)


gray = cv2.resize(
    gray,
    (
        GRID_SIZE,
        GRID_SIZE
    ),
    interpolation=cv2.INTER_CUBIC
)


height_map = (
    gray.astype(np.float32) / 255.0
)


# ============================================================
# 3D ВЫСОТА
# ============================================================

Z = (
    CARPET_THICKNESS
    + height_map * RELIEF_HEIGHT
)


# ============================================================
# X / Y
# ============================================================

rows, cols = Z.shape

aspect = rows / cols


x = np.linspace(
    0,
    1,
    cols
)

y = np.linspace(
    0,
    aspect,
    rows
)


X, Y = np.meshgrid(
    x,
    y
)


# ============================================================
# RGB ЦВЕТА
# ============================================================

vertex_colors = []

for r in range(rows):

    for c in range(cols):

        R = int(texture[r, c, 0])
        G = int(texture[r, c, 1])
        B = int(texture[r, c, 2])

        vertex_colors.append(
            f"rgb({R},{G},{B})"
        )


# ============================================================
# 3D
# ============================================================

st.subheader("3️⃣ Цветная 3D модель")

fig = go.Figure()


fig.add_trace(
    go.Mesh3d(

        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),

        vertexcolor=vertex_colors,

        flatshading=False,

        hoverinfo="skip"

    )
)


# ============================================================
# ВИД МОДЕЛИ
# ============================================================

fig.update_layout(

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    scene=dict(

        xaxis=dict(
            visible=False
        ),

        yaxis=dict(
            visible=False
        ),

        zaxis=dict(
            visible=False
        ),

        aspectmode="manual",

        # 🔵 ВИЗУАЛЬНО ОЧЕНЬ ТОНКАЯ МОДЕЛЬ
        aspectratio=dict(
            x=1,
            y=aspect,
            z=0.025
        ),

        camera=dict(
            eye=dict(
                x=1.4,
                y=1.4,
                z=0.8
            )
        )
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# СОЗДАНИЕ OBJ
# ============================================================

st.subheader("4️⃣ Скачать модель")


with tempfile.TemporaryDirectory() as temp_dir:

    obj_path = os.path.join(
        temp_dir,
        "tamga3d.obj"
    )

    mtl_path = os.path.join(
        temp_dir,
        "tamga3d.mtl"
    )

    texture_path = os.path.join(
        temp_dir,
        "carpet_texture.png"
    )

    zip_path = os.path.join(
        temp_dir,
        "Tamga3D_model.zip"
    )


    # ========================================================
    # ТЕКСТУРА
    # ========================================================

    cv2.imwrite(
        texture_path,
        cv2.cvtColor(
            original_crop,
            cv2.COLOR_RGB2BGR
        )
    )


    # ========================================================
    # OBJ
    # ========================================================

    with open(
        obj_path,
        "w",
        encoding="utf-8"
    ) as obj:

        obj.write(
            "mtllib tamga3d.mtl\n"
        )

        # ----------------------------------------------------
        # ВЕРХНИЕ ВЕРШИНЫ
        # ----------------------------------------------------

        for r in range(rows):

            for c in range(cols):

                obj.write(
                    f"v {X[r,c]:.6f} "
                    f"{Y[r,c]:.6f} "
                    f"{Z[r,c]:.6f}\n"
                )


        # ----------------------------------------------------
        # НИЖНИЕ ВЕРШИНЫ
        # ----------------------------------------------------

        bottom_offset = rows * cols


        for r in range(rows):

            for c in range(cols):

                obj.write(
                    f"v {X[r,c]:.6f} "
                    f"{Y[r,c]:.6f} "
                    f"0.000000\n"
                )


        # ----------------------------------------------------
        # UV
        # ----------------------------------------------------

        for r in range(rows):

            for c in range(cols):

                u = c / (cols - 1)

                v = 1 - (
                    r / (rows - 1)
                )

                obj.write(
                    f"vt {u:.6f} {v:.6f}\n"
                )


        # ----------------------------------------------------
        # ВЕРХНЯЯ ПОВЕРХНОСТЬ
        # ----------------------------------------------------

        obj.write(
            "usemtl CarpetTexture\n"
        )


        for r in range(rows - 1):

            for c in range(cols - 1):

                a = (
                    r * cols
                    + c
                    + 1
                )

                b = a + 1

                d = (
                    (r + 1) * cols
                    + c
                    + 1
                )

                e = d + 1


                obj.write(
                    f"f {a}/{a} "
                    f"{b}/{b} "
                    f"{e}/{e}\n"
                )


                obj.write(
                    f"f {a}/{a} "
                    f"{e}/{e} "
                    f"{d}/{d}\n"
                )


        # ----------------------------------------------------
        # НИЖНЯЯ ПОВЕРХНОСТЬ
        # ----------------------------------------------------

        obj.write(
            "usemtl CarpetSide\n"
        )


        for r in range(rows - 1):

            for c in range(cols - 1):

                a = (
                    bottom_offset
                    + r * cols
                    + c
                    + 1
                )

                b = a + 1

                d = (
                    bottom_offset
                    + (r + 1) * cols
                    + c
                    + 1
                )

                e = d + 1


                obj.write(
                    f"f {a} {e} {b}\n"
                )

                obj.write(
                    f"f {a} {d} {e}\n"
                )


        # ----------------------------------------------------
        # БОКОВЫЕ СТЕНКИ
        # ----------------------------------------------------

        # Верхняя сторона

        for c in range(cols - 1):

            top1 = c + 1
            top2 = c + 2

            bot1 = (
                bottom_offset
                + c
                + 1
            )

            bot2 = (
                bottom_offset
                + c
                + 2
            )

            obj.write(
                f"f {top1} {top2} {bot2}\n"
            )

            obj.write(
                f"f {top1} {bot2} {bot1}\n"
            )


        # Нижняя сторона

        for c in range(cols - 1):

            top1 = (
                (rows - 1)
                * cols
                + c
                + 1
            )

            top2 = top1 + 1

            bot1 = (
                bottom_offset
                + (rows - 1)
                * cols
                + c
                + 1
            )

            bot2 = bot1 + 1

            obj.write(
                f"f {top1} {bot2} {top2}\n"
            )

            obj.write(
                f"f {top1} {bot1} {bot2}\n"
            )


        # Левая сторона

        for r in range(rows - 1):

            top1 = (
                r * cols
                + 1
            )

            top2 = (
                (r + 1) * cols
                + 1
            )

            bot1 = (
                bottom_offset
                + r * cols
                + 1
            )

            bot2 = (
                bottom_offset
                + (r + 1) * cols
                + 1
            )

            obj.write(
                f"f {top1} {bot1} {top2}\n"
            )

            obj.write(
                f"f {top2} {bot1} {bot2}\n"
            )


        # Правая сторона

        for r in range(rows - 1):

            top1 = (
                r * cols
                + cols
            )

            top2 = (
                (r + 1) * cols
                + cols
            )

            bot1 = (
                bottom_offset
                + r * cols
                + cols
            )

            bot2 = (
                bottom_offset
                + (r + 1) * cols
                + cols
            )

            obj.write(
                f"f {top1} {top2} {bot1}\n"
            )

            obj.write(
                f"f {top2} {bot2} {bot1}\n"
            )


    # ========================================================
    # MTL
    # ========================================================

    with open(
        mtl_path,
        "w",
        encoding="utf-8"
    ) as mtl:

        mtl.write(
            "newmtl CarpetTexture\n"
        )

        mtl.write(
            "Ka 1.0 1.0 1.0\n"
        )

        mtl.write(
            "Kd 1.0 1.0 1.0\n"
        )

        mtl.write(
            "Ks 0.0 0.0 0.0\n"
        )

        mtl.write(
            "illum 1\n"
        )

        mtl.write(
            "map_Kd carpet_texture.png\n"
        )

        mtl.write("\n")

        mtl.write(
            "newmtl CarpetSide\n"
        )

        mtl.write(
            "Ka 0.25 0.18 0.10\n"
        )

        mtl.write(
            "Kd 0.25 0.18 0.10\n"
        )

        mtl.write(
            "Ks 0.0 0.0 0.0\n"
        )

        mtl.write(
            "illum 1\n"
        )


    # ========================================================
    # ZIP
    # ========================================================

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        archive.write(
            obj_path,
            "tamga3d.obj"
        )

        archive.write(
            mtl_path,
            "tamga3d.mtl"
        )

        archive.write(
            texture_path,
            "carpet_texture.png"
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    with open(
        zip_path,
        "rb"
    ) as file:

        st.download_button(
            "📦 Скачать Tamga3D",
            data=file,
            file_name="Tamga3D_model.zip",
            mime="application/zip",
            key="download_model",
            on_click="ignore"
        )


st.success(
    "✅ Модель готова: оригинальный цвет + "
    "тонкая основа + небольшой рельеф."
)
