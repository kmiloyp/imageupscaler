import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st
import replicate
import os
from urllib.request import urlopen
import base64
from PIL import ImageEnhance

@st.cache_resource
def load_model():
    """
    Carga y verifica la conexión con Replicate.
    """
    try:
        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            st.error("Token de Replicate no encontrado")
            return None

        if not token.startswith('r8_'):
            st.error("El formato del token no es válido. Debe comenzar con 'r8_'")
            return None

        try:
            client = replicate.Client(api_token=token)
            return client
        except replicate.exceptions.ReplicateError as e:
            st.error(f"Error de autenticación con Replicate: {str(e)}")
            return None
        except Exception as client_error:
            st.error(f"Error al conectar con Replicate: {str(client_error)}")
            return None

    except Exception as e:
        st.error(f"Error al inicializar el cliente de Replicate: {str(e)}")
        return None

def apply_fine_tuning(image, params):
    """
    Aplica ajustes finos a la imagen usando los parámetros especificados.

    Args:
        image: PIL.Image - Imagen a ajustar
        params: dict - Parámetros de ajuste fino
    """
    try:
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Crear un objeto ImageEnhancer para cada ajuste
        if params.get('sharpness', 1.0) != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(params['sharpness'])

        if params.get('contrast', 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(params['contrast'])

        if params.get('brightness', 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(params['brightness'])

        if params.get('color_balance', 1.0) != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(params['color_balance'])

        return image
    except Exception as e:
        st.error(f"Error al aplicar ajustes finos: {str(e)}")
        return image

def process_image(image, scale_factor, advanced_params=None):
    """
    Procesa la imagen usando el modelo de upscaling Clarity AI.

    Args:
        image: PIL.Image - Imagen a procesar
        scale_factor: int - Factor de escala (2 o 3)
        advanced_params: dict - Parámetros avanzados opcionales
            - face_enhance: bool - Mejorar rostros
            - denoise_level: int - Nivel de reducción de ruido (0-3)
            - output_format: str - Formato de salida ('png' o 'jpeg')
            - jpeg_quality: int - Calidad JPEG (60-100)
            - sharpness: float - Nivel de nitidez (0.0-2.0)
            - contrast: float - Nivel de contraste (0.0-2.0)
            - brightness: float - Nivel de brillo (0.0-2.0)
            - color_balance: float - Balance de color (0.0-2.0)
    """
    try:
        client = load_model()
        if client is None:
            st.error("No se pudo inicializar el cliente de Replicate")
            return None

        try:
            # Configurar parámetros avanzados
            params = {
                "face_enhance": advanced_params.get('face_enhance', True) if advanced_params else True,
                "scale": scale_factor,
                "noise_level": advanced_params.get('denoise_level', 1) if advanced_params else 1
            }

            # Guardar la imagen temporalmente
            temp_path = "temp_image.png"
            image.save(temp_path)

            st.info("Procesando imagen con Clarity AI...")

            # Usar el modelo con los parámetros configurados
            with open(temp_path, "rb") as file:
                output = client.run(
                    "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                    input={
                        "image": file,
                        **params
                    }
                )

            # Debug log
            st.write("Debug - Tipo de respuesta:", type(output))
            st.write("Debug - Contenido de respuesta:", output)

            # Eliminar el archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

            try:
                # Obtener la URL de la imagen procesada
                if hasattr(output, 'url'):
                    image_url = output.url
                elif isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                elif isinstance(output, str):
                    image_url = output
                else:
                    st.error(f"Error: Formato de respuesta inesperado del modelo - {type(output)}")
                    return None

                st.info("Descargando imagen procesada...")
                response = urlopen(image_url)
                result_image = Image.open(response)

                # Aplicar ajustes finos si se especifican
                if advanced_params:
                    result_image = apply_fine_tuning(result_image, advanced_params)

                # Aplicar formato de salida si se especifica
                if advanced_params and advanced_params.get('output_format'):
                    output_format = advanced_params['output_format'].upper()
                    if output_format == 'JPEG':
                        if result_image.mode in ('RGBA', 'LA'):
                            result_image = result_image.convert('RGB')
                        buffer = io.BytesIO()
                        result_image.save(buffer, format='JPEG', 
                                        quality=advanced_params.get('jpeg_quality', 95))
                        buffer.seek(0)
                        result_image = Image.open(buffer)

                st.success("✅ Imagen procesada exitosamente")
                return result_image

            except Exception as e:
                st.error(f"Error al procesar la respuesta del modelo: {str(e)}")
                return None

        except replicate.exceptions.ReplicateError as api_error:
            st.error(f"Error al procesar la imagen: {str(api_error)}")
            return None

    except Exception as e:
        st.error(f"Error general: {str(e)}")
        return None

def get_image_download_link(img, filename, text):
    """Genera un link de descarga para la imagen procesada."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    b64 = base64.b64encode(img_str).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{text}</a>'
    return href