import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
import tempfile
import os
import zipfile

st.set_page_config(page_title="Tamga3D", layout="wide")

st.title("🧿 Tamga3D")
st.write("Выбери 4 угла орнамента на изображении.")

# =========================
# НАСТРОЙКИ МОДЕЛИ
# =========================

# 🔵 ТОЛЩИНА НАСТОЯЩЕГО КОВРА
CARPET_THICKNESS = 0.008

# 🟢 ВЫСОТА РЕЛЬЕФА ОРНАМЕНТА
RELIEF_HEIGHT = 0.004

# 🟡 РАЗМЕР СЕТКИ
GRID_SIZE = 180

# =========================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# =========================

uploaded_file = st.file_uploader(
    "Загрузи фотографию ковра",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("1. Выбери орнамент")

    st.write(
        "Нажми на 4 точки: "
        "левый верх → правый верх → правый низ → левый низ."
    )

    coords = []

    clicked = streamlit_image_coordinates(image_rgb)

    if clicked:
        coords.append((clicked["x"], clicked["y"]))

    if len(coords) == 1:
        st.info("Точка 1 выбрана. Теперь нажми следующую.")

    # Для выбора 4 точек используем состояние
    if "points" not in st.session_state:
        st.session_state.points = []

    if clicked:
        point = (clicked["x"], clicked["y"])

        if not st.session_state.points or point != st.session_state.points[-1]:
            st.session_state.points.append(point)

    points = st.session_state.points

    st.write("Выбрано точек:", len(points))

    if len(points) >= 4:

        # Берём первые 4 точки
        p1, p2, p3, p4 = points[:4]

        x1 = min(p1[0], p4[0])
        x2 = max(p2[0], p3[0])

        y1 = min(p1[1], p2[1])
        y2 = max(p3[1], p4[1])

        # Проверка
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_rgb.shape[1], x2)
        y2 = min(image_rgb.shape[0], y2)

        crop = image_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            st.error("Не удалось выделить область.")
            st.stop()

        st.subheader("2. Выбранный орнамент")

        st.image(crop, use_container_width=True)

        # =========================
        # ПОДГОТОВКА ЦВЕТА
        # =========================

        original_crop = crop.copy()

        # Сохраняем оригинальную текстуру
        texture = cv2.resize(
            original_crop,
            (GRID_SIZE, GRID_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # =========================
        # КАРТА РЕЛЬЕФА
        # =========================

        gray = cv2.cvtColor(
            original_crop,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.resize(
            gray,
            (GRID_SIZE, GRID_SIZE),
            interpolation=cv2.INTER_CUBIC
        )

        # Нормализация
        height_map = gray.astype(np.float32) / 255.0

        # Очень маленький рельеф
        Z_top = (
            CARPET_THICKNESS
            + height_map * RELIEF_HEIGHT
        )

        Z_bottom = np.zeros_like(Z_top)

        # =========================
        # СОЗДАНИЕ КООРДИНАТ
        # =========================

        rows, cols = Z_top.shape

        aspect = rows / cols

        X = np.linspace(
            0,
            1,
            cols
        )

        Y = np.linspace(
            0,
            aspect,
            rows
        )

        X, Y = np.meshgrid(X, Y)

        # =========================
        # ЦВЕТА
        # =========================

        vertex_colors = []

        for r in range(rows):
            for c in range(cols):

                R = int(texture[r, c, 0])
                G = int(texture[r, c, 1])
                B = int(texture[r, c, 2])

                vertex_colors.append(
                    f"rgb({R},{G},{B})"
                )

        # =========================
        # 3D ВИЗУАЛИЗАЦИЯ
        # =========================

        st.subheader("3. 3D модель")

        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z_top.flatten(),

                intensity=height_map.flatten(),

                vertexcolor=vertex_colors,

                alphahull=0,

                flatshading=False,

                name="Carpet"
            )
        )

        # =========================
        # НАСТРОЙКА ВИДА
        # =========================

        fig.update_layout(

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

                # 🔵 ДЕЛАЕМ КОВЁР ВИЗУАЛЬНО ТОНКИМ
                aspectratio=dict(
                    x=1,
                    y=aspect,
                    z=0.035
                ),

                camera=dict(
                    eye=dict(
                        x=1.4,
                        y=1.4,
                        z=0.7
                    )
                )
            ),

            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================
        # СОЗДАНИЕ OBJ
        # =========================

        st.subheader("4. Скачать модель")

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

            # Сохраняем оригинальный цвет
            cv2.imwrite(
                texture_path,
                cv2.cvtColor(
                    original_crop,
                    cv2.COLOR_RGB2BGR
                )
            )

            with open(
                obj_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "mtllib tamga3d.mtl\n"
                )

                # -------------------------
                # ВЕРХНИЕ ВЕРШИНЫ
                # -------------------------

                for r in range(rows):

                    for c in range(cols):

                        f.write(
                            f"v {X[r,c]:.6f} "
                            f"{Y[r,c]:.6f} "
                            f"{Z_top[r,c]:.6f}\n"
                        )

                # -------------------------
                # НИЖНИЕ ВЕРШИНЫ
                # -------------------------

                bottom_offset = rows * cols

                for r in range(rows):

                    for c in range(cols):

                        f.write(
                            f"v {X[r,c]:.6f} "
                            f"{Y[r,c]:.6f} "
                            f"{Z_bottom[r,c]:.6f}\n"
                        )

                # -------------------------
                # UV
                # -------------------------

                for r in range(rows):

                    for c in range(cols):

                        u = c / (cols - 1)
                        v = 1 - r / (rows - 1)

                        f.write(
                            f"vt {u:.6f} {v:.6f}\n"
                        )

                f.write("\n")
                f.write("usemtl CarpetTexture\n")

                # -------------------------
                # ВЕРХ
                # -------------------------

                for r in range(rows - 1):

                    for c in range(cols - 1):

                        a = r * cols + c + 1
                        b = a + 1
                        d = (r + 1) * cols + c + 1
                        e = d + 1

                        f.write(
                            f"f {a}/{a} {b}/{b} {e}/{e}\n"
                        )

                        f.write(
                            f"f {a}/{a} {e}/{e} {d}/{d}\n"
                        )

                # -------------------------
                # НИЗ
                # -------------------------

                f.write("\n")
                f.write("usemtl CarpetSide\n")

                for r in range(rows - 1):

                    for c in range(cols - 1):

                        a = bottom_offset + r * cols + c + 1
                        b = a + 1
                        d = bottom_offset + (r + 1) * cols + c + 1
                        e = d + 1

                        f.write(
                            f"f {a} {e} {b}\n"
                        )

                        f.write(
                            f"f {a} {d} {e}\n"
                        )

                # -------------------------
                # БОКОВЫЕ СТЕНКИ
                # -------------------------

                # Верхняя сторона

                for c in range(cols - 1):

                    a = c + 1
                    b = c + 2

                    c2 = bottom_offset + c + 1
                    d2 = bottom_offset + c + 2

                    f.write(
                        f"f {a} {b} {d2}\n"
                    )

                    f.write(
                        f"f {a} {d2} {c2}\n"
                    )

                # Нижняя сторона

                for c in range(cols - 1):

                    a = (rows - 1) * cols + c + 1
                    b = a + 1

                    c2 = bottom_offset + (rows - 1) * cols + c + 1
                    d2 = c2 + 1

                    f.write(
                        f"f {a} {d2} {b}\n"
                    )

                    f.write(
                        f"f {a} {c2} {d2}\n"
                    )

                # Левая сторона

                for r in range(rows - 1):

                    a = r * cols + 1
                    b = (r + 1) * cols + 1

                    c2 = bottom_offset + r * cols + 1
                    d2 = bottom_offset + (r + 1) * cols + 1

                    f.write(
                        f"f {a} {c2} {b}\n"
                    )

                    f.write(
                        f"f {b} {c2} {d2}\n"
                    )

                # Правая сторона

                for r in range(rows - 1):

                    a = r * cols + cols
                    b = (r + 1) * cols + cols

                    c2 = bottom_offset + r * cols + cols
                    d2 = bottom_offset + (r + 1) * cols + cols

                    f.write(
                        f"f {a} {b} {c2}\n"
                    )

                    f.write(
                        f"f {b} {d2} {c2}\n"
                    )

            # =========================
            # MTL
            # =========================

            with open(
                mtl_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "newmtl CarpetTexture\n"
                )

                f.write(
                    "Ka 1.0 1.0 1.0\n"
                )

                f.write(
                    "Kd 1.0 1.0 1.0\n"
                )

                f.write(
                    "Ks 0.0 0.0 0.0\n"
                )

                f.write(
                    "illum 1\n"
                )

                f.write(
                    "map_Kd carpet_texture.png\n"
                )

                f.write("\n")

                f.write(
                    "newmtl CarpetSide\n"
                )

                f.write(
                    "Ka 0.3 0.2 0.1\n"
                )

                f.write(
                    "Kd 0.3 0.2 0.1\n"
                )

                f.write(
                    "Ks 0.0 0.0 0.0\n"
                )

                f.write(
                    "illum 1\n"
                )

            # =========================
            # ZIP
            # =========================

            zip_path = os.path.join(
                temp_dir,
                "Tamga3D_model.zip"
            )

            with zipfile.ZipFile(
                zip_path,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zipf:

                zipf.write(
                    obj_path,
                    "tamga3d.obj"
                )

                zipf.write(
                    mtl_path,
                    "tamga3d.mtl"
                )

                zipf.write(
                    texture_path,
                    "carpet_texture.png"
                )

            with open(
                zip_path,
                "rb"
            ) as file:

                st.download_button(
                    label="📦 Скачать 3D модель",
                    data=file,
                    file_name="Tamga3D_model.zip",
                    mime="application/zip"
                )

        st.success(
            "Готово! Модель содержит цветную текстуру, "
            "тонкую основу и небольшой рельеф."
        )

else:

    st.info(
        "Сначала загрузи фотографию ковра."
    )
