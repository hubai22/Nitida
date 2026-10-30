# Nítida — mejora de fotos con IA para iPhone

App nativa para iPhone (iOS 16 o superior) que mejora fotos de baja calidad:

1. **Toda la foto**: Real-ESRGAN limpia artefactos de compresión, ruido y desenfoque, y la amplía ×4 (fondo, pelo, ropa, texto).
2. **Las caras**: SCRFD localiza cada cara y GFPGAN reconstruye ojos, piel, dientes y pelo con detalle nuevo (no es un filtro de nitidez).
3. Comparador antes/después, control de intensidad en caras, guardar en Fotos y compartir.

Todo se procesa dentro del iPhone; las fotos no se suben a ningún servidor. La primera vez la app descarga el motor de IA (unos 380 MB, mejor con Wi-Fi).

---

## Paso 1 — Compilar la app (sin Mac)

GitHub compila la app gratis en una Mac en la nube y te da el archivo `Nitida.ipa`.

1. Crea una cuenta gratis en https://github.com (se puede desde el iPhone).
2. Crea un repositorio **vacío** llamado `nitida` (botón **New repository**). Puede ser público o privado.
3. Sube el proyecto. Dos formas:
   - **La fácil:** conecta GitHub a Claude (claude.ai › Configuración › Conectores › GitHub) y dile a Claude el nombre del repositorio. Claude sube el código, lanza la compilación y corrige cualquier error que aparezca.
   - **A mano (desde una computadora):** en el repositorio, **Add file › Upload files** y arrastra el contenido de esta carpeta. Las carpetas que empiezan con punto no siempre se suben así, así que luego entra a **Add file › Create new file**, escribe como nombre `.github/workflows/build.yml` y pega el contenido del archivo `workflow-build.yml` incluido aquí.
4. Ve a la pestaña **Actions**: verás "Compilar app para iPhone" en marcha (unos 10 minutos).
5. Cuando termine con ✓, entra a **Releases** (en la página principal del repositorio) y descarga `Nitida.ipa`.

## Paso 2 — Instalarla en tu iPhone (sin App Store)

Se necesita **una computadora una sola vez** (Windows, Mac o Linux, puede ser prestada) y tu Apple ID normal y gratis.

En el iPhone:
1. Pon un código de desbloqueo (si no tienes).
2. **Ajustes › Privacidad y seguridad › Modo de desarrollador** → activar (el iPhone se reinicia).

En la computadora:
3. Descarga **iloader** desde https://iloader.app o https://github.com/nab138/iloader (son las únicas fuentes oficiales). En Windows instala antes iTunes.
4. Conecta el iPhone por cable, acepta "Confiar en esta computadora", abre iloader e inicia sesión con tu Apple ID.
5. Elige **Install SideStore** (o "LiveContainer + SideStore"). Tarda unos minutos.

De vuelta en el iPhone (ya sin computadora):
6. **Ajustes › General › VPN y gestión de dispositivos** → confía en tu Apple ID.
7. Instala **LocalDevVPN** desde el App Store y actívalo.
8. Abre SideStore, toca **+** y elige `Nitida.ipa` (pásalo al iPhone por AirDrop, iCloud Drive o descárgalo directo de Releases en Safari).

### Renovación cada 7 días

Con un Apple ID gratis, Apple hace que las apps instaladas así caduquen a los 7 días. Para renovarla **desde el propio iPhone**: activa LocalDevVPN, abre SideStore y toca **Refresh All**. Puedes automatizarlo con un atajo de la app Atajos.

---

## Uso

1. Abre Nítida y toca **Descargar** (solo la primera vez).
2. **Elegir foto** › **Mejorar foto**. Una foto típica tarda entre 20 segundos y 2 minutos según el modelo de iPhone.
3. Arrastra el control del centro para comparar antes/después, ajusta **Intensidad en caras** y toca **Guardar**.

Si alguna foto sale rara o la app se cierra al mejorar, desactiva **Ajustes (⚙︎) › Acelerar con Neural Engine**.

## Detalles técnicos

- SwiftUI + ONNX Runtime 1.24 (Core ML para detector y ampliación; la cara va en CPU para evitar errores de precisión).
- Fotos de más de 1024 px se reducen antes de ampliar ×4 (salida máxima ~4096 px).
- Proyecto generado con XcodeGen (`project.yml`); compilación en `.github/workflows/build.yml`.

## Licencias de los modelos

GFPGAN (Apache 2.0), Real-ESRGAN (BSD-3), SCRFD/InsightFace (uso no comercial). Esta app es para uso personal.
