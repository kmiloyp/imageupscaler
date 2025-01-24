import streamlit as st
from PIL import Image
import io
from utils import process_image, get_image_download_link, load_model
import os
from datetime import datetime

def check_token():
    """Verifica el token de acceso."""
    session_token = st.session_state.get('access_token', '')
    correct_token = os.environ.get('APP_ACCESS_TOKEN')

    if not correct_token:
        st.error("Error de configuración: Token de acceso no configurado en el sistema")
        return False

    if not session_token:
        token_input = st.sidebar.text_input("🔐 Token de acceso", type="password")
        if st.sidebar.button("Verificar acceso"):
            if token_input == correct_token:
                st.session_state.access_token = token_input
                return True
            else:
                st.sidebar.error("Token inválido")
                return False
        return False

    return session_token == correct_token

def initialize_session_state():
    """Inicializa las variables de estado de la sesión"""
    if 'processed_images_history' not in st.session_state:
        st.session_state.processed_images_history = []

def main():
    initialize_session_state()

    # Configuración de la página
    st.set_page_config(
        page_title="Mejorador de Imágenes con AI",
        page_icon="🖼️",
        layout="wide"
    )

    # Verificar acceso
    if not check_token():
        st.warning("Por favor ingrese el token de acceso para usar la aplicación")
        return

    # Mostrar logo
    logo = Image.open('attached_assets/iflexo6-final.png')
    st.image(logo, width=200)

    st.title("Mejorador de Imágenes con AI")
    st.write("Sube una imagen y mejora su calidad usando tecnología avanzada de interpolación e IA")

    # Tabs principales de la aplicación
    tab_proces, tab_history = st.tabs(["Procesamiento", "Historial"])

    with tab_proces:
        # Sección de estado del sistema
        with st.sidebar:
            st.header("Estado del Sistema")
            api_status = st.empty()

            # Verificar el token y mostrar el estado
            token = os.environ.get("REPLICATE_API_TOKEN")
            if not token:
                api_status.error("❌ Error en la conexión con la API")
                st.error("El token de Replicate no está configurado")
                st.warning("""
                Para usar esta herramienta, necesitas configurar el token de Replicate.
                Por favor, contacta al administrador del sistema.
                """)
            else:
                # Intentar cargar el modelo para verificar la conexión
                with st.spinner("Verificando conexión con la API..."):
                    client = load_model()
                    if client is not None:
                        api_status.success("✅ API conectada y funcionando")
                    else:
                        api_status.error("❌ Error en la conexión con la API")

        # Botón de cierre de sesión
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.access_token = ""
            st.experimental_rerun()


        # Subida de archivo
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (JPG o PNG)",
            type=["jpg", "jpeg", "png"],
            help="Límite 200MB por file - JPG, JPEG, PNG"
        )

        # Configuración básica
        col1, col2 = st.columns(2)
        with col1:
            scale_factor = st.select_slider(
                "Factor de escala",
                options=[2, 3],
                value=2,
                help="Factor por el cual se aumentará la resolución de la imagen"
            )

        with col2:
            upscale_method = st.selectbox(
                "Método de upscaling",
                options=["✨ Clarity AI (Mejor Calidad)"],
                help="Clarity AI ofrece los mejores resultados para la mayoría de las imágenes"
            )

        # Opciones avanzadas
        with st.expander("⚙️ Opciones Avanzadas"):
            st.info("Estas opciones permiten un control más preciso sobre el proceso de mejora")

            advanced_col1, advanced_col2 = st.columns(2)

            with advanced_col1:
                face_enhance = st.toggle(
                    "Mejorar rostros",
                    value=True,
                    help="Aplica mejoras específicas para rostros en la imagen"
                )

                denoise_level = st.slider(
                    "Nivel de reducción de ruido",
                    min_value=0,
                    max_value=3,
                    value=1,
                    help="Mayor valor = más suavizado, menor valor = más detalle"
                )

            with advanced_col2:
                output_format = st.selectbox(
                    "Formato de salida",
                    options=["PNG", "JPEG"],
                    index=0,
                    help="PNG mantiene mejor calidad pero genera archivos más grandes"
                )

                if output_format == "JPEG":
                    jpeg_quality = st.slider(
                        "Calidad JPEG",
                        min_value=60,
                        max_value=100,
                        value=95,
                        help="Mayor valor = mejor calidad pero archivo más grande"
                    )

        # Modo de ajuste fino
        with st.expander("🎨 Modo de Ajuste Fino"):
            st.info("Ajusta con precisión los parámetros de calidad de la imagen")

            fine_tune_col1, fine_tune_col2 = st.columns(2)

            with fine_tune_col1:
                sharpness = st.slider(
                    "Nitidez",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ajusta la nitidez de los detalles en la imagen"
                )

                contrast = st.slider(
                    "Contraste",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ajusta la diferencia entre claros y oscuros"
                )

            with fine_tune_col2:
                brightness = st.slider(
                    "Brillo",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ajusta la luminosidad general de la imagen"
                )

                color_balance = st.slider(
                    "Balance de Color",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ajusta la intensidad de los colores"
                )

        # Procesamiento de imagen
        if uploaded_file:

            # Leer imagen original
            image = Image.open(uploaded_file)

            # Verificar tamaño de imagen
            file_size = uploaded_file.size / (1024 * 1024)  # Convertir a MB
            if file_size > 200:
                st.error("❌ El archivo es demasiado grande. Por favor, usa una imagen menor a 200MB.")
                return

            # Crear columnas para comparación side-by-side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagen Original")
                st.image(image, use_container_width=True)
                st.info(f"Dimensiones originales: {image.size[0]}x{image.size[1]} px")

            if st.button("Procesar Imagen", type="primary"):
                with st.spinner("Procesando imagen con IA... Esto puede tomar unos momentos."):
                    # Crear diccionario de parámetros avanzados
                    advanced_params = {
                        "face_enhance": face_enhance,
                        "denoise_level": denoise_level,
                        "output_format": output_format.lower(),
                        "jpeg_quality": jpeg_quality if output_format == "JPEG" else None,
                        # Agregar parámetros de ajuste fino
                        "sharpness": sharpness,
                        "contrast": contrast,
                        "brightness": brightness,
                        "color_balance": color_balance
                    }

                    processed_image = process_image(image, scale_factor, advanced_params)

                if processed_image:
                    with col2:
                        st.subheader("Imagen Mejorada")
                        st.image(processed_image, use_container_width=True)
                        st.info(f"Nuevas dimensiones: {processed_image.size[0]}x{processed_image.size[1]} px")

                        # Botón de descarga
                        download_filename = f"mejorada_{uploaded_file.name}"
                        st.markdown(
                            get_image_download_link(processed_image, download_filename, "📥 Descargar imagen mejorada"),
                            unsafe_allow_html=True
                        )

                    # Guardar en el historial
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    history_entry = {
                        "timestamp": timestamp,
                        "original_image": image,
                        "processed_image": processed_image,
                        "original_name": uploaded_file.name,
                        "parameters": {
                            "scale_factor": scale_factor,
                            "face_enhance": face_enhance,
                            "denoise_level": denoise_level,
                            "output_format": output_format,
                            "sharpness": sharpness,
                            "contrast": contrast,
                            "brightness": brightness,
                            "color_balance": color_balance
                        }
                    }
                    st.session_state.processed_images_history.append(history_entry)


    with tab_history:
        st.header("Historial de Procesamiento")
        if not st.session_state.processed_images_history:
            st.info("No hay imágenes procesadas en el historial")
        else:
            for idx, entry in enumerate(reversed(st.session_state.processed_images_history)):
                with st.expander(f"Imagen {entry['original_name']} - {entry['timestamp']}"):
                    hist_col1, hist_col2 = st.columns(2)

                    with hist_col1:
                        st.subheader("Original")
                        st.image(entry['original_image'], use_container_width=True)

                    with hist_col2:
                        st.subheader("Mejorada")
                        st.image(entry['processed_image'], use_container_width=True)

                    # Mostrar parámetros usados
                    st.markdown("**Parámetros utilizados:**")
                    st.json(entry['parameters'])

    # Información adicional
    with st.expander("ℹ️ Acerca de esta herramienta"):
        st.markdown("""
        ### Información sobre el proceso de mejora

        Esta herramienta utiliza Clarity AI, un modelo de inteligencia artificial especializado
        en mejorar la calidad y resolución de imágenes. El proceso incluye:

        1. **Análisis de la imagen**: El modelo analiza los detalles y patrones
        2. **Mejora de resolución**: Aumenta el tamaño manteniendo la calidad
        3. **Optimización**: Mejora la nitidez y los detalles finos

        ### Recomendaciones:
        - Para mejores resultados, usa imágenes de resolución media
        - Evita imágenes demasiado pequeñas o muy grandes
        - El tiempo de procesamiento depende del tamaño de la imagen
        """)

if __name__ == "__main__":
    main()