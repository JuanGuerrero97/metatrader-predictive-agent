# MetaTrader Predictive Trading Assistant

Este proyecto implementa un modelo de predicci\u00f3n para ayudar a inversionistas a decidir si comprar o vender activos financieros utilizando datos hist\u00f3ricos de **MetaTrader**. El proyecto est\u00e1 dise\u00f1ado con principios \u00e9ticos y sostenibles, promoviendo la transparencia y el manejo responsable del riesgo.

## Objetivos

- **Predicci\u00f3n de decisiones de compra/venta**: entrenar un modelo que aprenda patrones en los datos de mercado para sugerir oportunidades de trading (compra o venta) en funci\u00f3n de indicadores t\u00e9cnicos.
- **Interfaz gr\u00e1fica para inversionistas**: ofrecer una interfaz intuitiva (con Streamlit) donde los usuarios puedan cargar datos, visualizar gr\u00e1ficos, ejecutar el modelo y recibir recomendaciones con explicaciones.
- **\u00c9tica y sostenibilidad**: incorporar pr\u00e1cticas \u00e9ticas de inversi\u00f3n, como advertencias de riesgo, promoci\u00f3n de estrategias de inversi\u00f3n a largo plazo y respeto por la privacidad de los datos. El asistente no garantiza ganancias y anima al usuario a considerar aspectos ESG (ambientales, sociales y de gobierno) al tomar decisiones.

## Estructura del repositorio

```
metatrader_predictive_agent/
├── README.md             # Esta gu\u00eda
├── requirements.txt      # Dependencias del proyecto
├── data/                 # Lugar para datos brutos de MetaTrader (no se incluye en el repositorio)
├── notebooks/
│   └── exploratory.ipynb # Ejemplo de an\u00e1lisis exploratorio (para uso local)
├── src/
│   ├── __init__.py
│   ├── data_utils.py     # Funciones para cargar y preparar datos de MetaTrader
│   ├── model.py          # Definici\u00f3n y entrenamiento del modelo predictivo
│   ├── interface.py      # Interfaz Streamlit para usuarios
│   └── utils.py          # Utilidades comunes
└── LICENSE               # Licencia de c\u00f3digo abierto (MIT)
```

## Uso

1. **Instalaci\u00f3n de dependencias**

   Antes de ejecutar el c\u00f3digo, crea un entorno virtual (opcional) e instala las dependencias:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Preparaci\u00f3n de datos**

   Coloca tus archivos de datos de MetaTrader (por ejemplo, archivos CSV exportados con columnas `time`, `open`, `high`, `low`, `close`, `volume`) en el directorio `data/`. Aseg\u00farate de que los nombres de las columnas sean consistentes.

3. **Entrenamiento del modelo**

   Ejecuta el script de entrenamiento para generar un modelo almacenado en `models/`:

   ```bash
   python src/model.py --train --data_path data/tu_archivo.csv
   ```

   Este script realizar\u00e1 un an\u00e1lisis de las series de tiempo y entrenar\u00e1 un modelo (p.ej., **RandomForestClassifier** o **XGBoost**) para predecir se\u00f1ales de compra/venta basadas en indicadores t\u00e9cnicos como medias m\u00f3viles, RSI u otros. El modelo se guardar\u00e1 como un archivo `joblib` para uso posterior.

4. **Ejecuci\u00f3n de la interfaz**

   Inicia la interfaz Streamlit para interactuar con el modelo y visualizar gr\u00e1ficas:

   ```bash
   streamlit run src/interface.py
   ```

   La aplicaci\u00f3n permitir\u00e1 cargar datos de mercado recientes, mostrar indicadores calculados y generar una recomendaci\u00f3n de compra/venta con explicaciones. Tambi\u00e9n proporcionar\u00e1 estad\u00edsticas de rendimiento y advertencias sobre riesgos.

## Consideraciones \u00e9ticas y sostenibles

- **Transparencia**: El c\u00f3digo abierto permite a otros auditar la l\u00f3gica del modelo. Adem\u00e1s, se proporcionan m\u00e9tricas de desempe\u00f1o para evaluar la confiabilidad.
- **Advertencia de riesgo**: Las recomendaciones son probabil\u00edsticas y no deben considerarse asesoramiento financiero. Se alienta a los usuarios a diversificar sus inversiones y considerar asesor\u00eda profesional.
- **Sostenibilidad**: Se anima a usar el modelo para apoyar decisiones que consideren factores ambientales, sociales y de buen gobierno (ESG). Aunque el modelo est\u00e1 centrado en patrones t\u00e9cnicos, el usuario debe integrar perspectivas sostenibles en su estrategia.
- **Privacidad**: Los datos de los usuarios no se almacenan en servidores externos; el procesamiento se realiza localmente.

## Licencia

Este proyecto est\u00e1 licenciado bajo la licencia MIT. Consulta el archivo `LICENSE` para m\u00e1s detalles.
