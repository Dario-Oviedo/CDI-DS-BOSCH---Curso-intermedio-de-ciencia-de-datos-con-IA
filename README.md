# CDI DS BOSCH - Curso intermedio de ciencia de datos con IA
CDI DS BOSCH - Curso intermedio de ciencia de datos con IA
<br>
<br>
## Introducción
Durante este curso, aprendimos los principios basicos para la implementación de los algoritmos de aprendizaje supervisado. <br>
Entre los principales algoritmos tenemos KNN, Random forest, SVM. <br>
<br>
## Proyecto Final
El proyecto final consiste en generar un repositorio y subir el archivo proyecto final.<br>
En este archivo, Readme, trato de explicar las diferentes secciones del archivo.<br>

### Obtención de datos crudos
En esta parte, obtenemos los datos de la Secretaria de Salud:
<br><br>
<em>
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git<br>
!python rapidsai-csp-utils/colab/pip-install.py<br>
!wget https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip<br>
!unzip /content/datos_abiertos_covid19.zip<br>
</em>
<br>
<br>
### Import Section
Esto lo utilizo en clara alusión a la manera que se usa en Cobol. A titulo personal, considero que tener todo agrupado ayuda a generar un código de fácil mantenimiento en el futuro.
<br><br>
<em>
import pandas as pd<br>
import numpy as np<br>
from sklearn.preprocessing import LabelEncoder<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.metrics import classification_report<br>
from sklearn.ensemble import RandomForestClassifier<br>
import locale<br>
</em>
<br>
<br>
### Generación de Dataframe
En esta sección podemos revisar como se genera el Dataframe a partir de los datos obtenidos.
<br><br>
<em>
def getpreferredencoding(do_setlocale = True):<br>
&emsp;  return "UTF-8" <br>
<br>
locale.getpreferredencoding = getpreferredencoding<br>
data = pd.read_csv('COVID19MEXICO.csv', low_memory=False)<br>
data<br>
</em>
<br>
<br>
### Backup y eliminación de columnas no usadas.
Debido al tamaño de los datos, es necesario remover algunas columnas. Antes de realizar esta acción, es conveniente mantener una copia del Dataframe
<br><br>
<em>
df = data
<br>
df = df.drop(columns=['SECTOR','MUNICIPIO_RES','ID_REGISTRO','ORIGEN','PAIS_NACIONALIDAD','PAIS_ORIGEN','INDIGENA','ENTIDAD_UM','ENTIDAD_NAC','NACIONALIDAD','ENTIDAD_RES'])
</em>
<br>
<br>
### Formateo de datos
En esta sección se formatean las columnas.
<br><br>
<em>
df['FECHA_ACTUALIZACION'] = pd.to_datetime(df['FECHA_ACTUALIZACION'])<br>
df['FECHA_INGRESO'] = pd.to_datetime(df['FECHA_INGRESO'])<br>
df['FECHA_SINTOMAS'] = pd.to_datetime(df['FECHA_SINTOMAS'])<br>
df['FECHA_DEF'] = pd.to_datetime(df['FECHA_DEF'], errors = 'coerce')
</em>
<br>
<br>
### Remover columnas que no son int64
Se eliminan las columnas que no se pueden convertir en int64
<br><br>
<em>
df = df.drop(columns=['FECHA_ACTUALIZACION', 'FECHA_INGRESO', 'FECHA_SINTOMAS', 'FECHA_DEF','DIAS_DEFUNCION', 'DIAS_SINTOMAS'])
</em>
<br>
<br>

### Generación de X y Y
Se generan X y Y para el estudio
<br><br>
<em>
X = df.drop(columns = ['CLASIFICACION_FINAL'])<br>
y = df['CLASIFICACION_FINAL']<br>
<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br>
<br>
clf = RandomForestClassifier(random_state=42)<br>
clf.fit(X_train, y_train)<br>
y_pred = clf.predict(X_test)<br>
<br>
clf = RandomForestClassifier(random_state=42)<br>
clf.fit(X_train, y_train)<br>
y_pred = clf.predict(X_test)<br>
<br>
print(classification_report(y_test, y_pred))<br>
</em>
<br>
<br>
### CUML
Uso de CUML para reducir el tiempo de espera en los calculos
<br><br>
<em>
import cuml<br>
from cuml.ensemble import RandomForestClassifier<br>
from sklearn.metrics import classification_report<br>
<br>
clf = RandomForestClassifier(random_state=42)<br>
<br>
X_train = X_train.astype('float32')<br>
y_train = y_train.astype('float32')<br>
<br>
clf.fit(X_train, y_train)<br>
y_pred = clf.predict(X_test)<br>
<br>
print(classification_report(y_test, y_pred))<br>
</em>
<br>
<br>
### GRADIO
Uso de GRADIO para generar una API
<br><br>
<em>
import gradio as gr<br>
import pandas as pd<br>
import numpy as np<br>
<br>

def predict(sexo, tipo_paciente, intubado, neumonia, edad, embarazo,<br>
&emsp;&emsp;             diabetes, epoc, asma, inmusupr,<br>
&emsp;&emsp;            hipertension, otra_com, cardiovascular, obesidad,<br>
&emsp;&emsp;            renal_cronica, tabaquismo, otro_caso, toma_muestra_lab,<br>
&emsp;&emsp;            resultado_lab, toma_muestra_antigeno, resultado_antigeno,<br>
&emsp;&emsp;           uci, dias_hospitalizacion, dias_sintomas):<br>
<br>
&emsp;    input_data = pd.DataFrame([[<br>
&emsp;&emsp;        sexo, tipo_paciente, intubado, neumonia, edad, embarazo,<br>
&emsp;&emsp;       diabetes, epoc, asma, inmusupr,<br>
&emsp;&emsp;        hipertension, otra_com, cardiovascular, obesidad,<br>
&emsp;&emsp;        renal_cronica, tabaquismo, otro_caso, toma_muestra_lab,<br>
&emsp;&emsp;        resultado_lab, toma_muestra_antigeno, resultado_antigeno, uci, dias_hospitalizacion, dias_sintomas<br>
&emsp;&emsp;        ]], columns=[<br>
&emsp;&emsp;        'SEXO', 'TIPO_PACIENTE', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO','DIABETES', 'EPOC',<br>
&emsp;&emsp;        'ASMA', 'INMUSUPR','HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD','RENAL_CRONICA',<br>
&emsp;&emsp;        'TABAQUISMO', 'OTRO_CASO', 'TOMA_MUESTRA_LAB','RESULTADO_LAB', 'TOMA_MUESTRA_ANTIGENO',<br>
&emsp;&emsp;        'RESULTADO_ANTIGENO','UCI', 'DIAS_HOSPITALIZACION', 'DIAS_SINTOMAS'])<br>
<br>
    prediction = clf.predict(input_data)<br>
    return prediction<br>



<br>
gr.Row<br>
inputs = [<br>
&emsp;    gr.Radio(choices=[0, 1], label='Sexo'),<br>
&emsp;    gr.Radio(choices=[0, 1], label='Tipo de Paciente'),<br>
&emsp;    gr.Radio(choices=[0, 1, 97, 98, 99], label='Intubado'),<br>
&emsp;    gr.Radio(choices=[0, 1, 97, 98, 99], label='Neumonía'),<br>
&emsp;    gr.Slider(minimum=0, maximum=120, value=30, label='Edad'),<br>
&emsp;    gr.Radio(choices=[1, 2, 97, 98, 99], label='Embarazo'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Diabetes'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='EPOC'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Asma'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Inmunosupresión'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Hipertensión'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Otra Comorbilidad'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Cardiovascular'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Obesidad'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Enfermedad Renal Crónica'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Tabaquismo'),<br>
&emsp;    gr.Radio(choices=[1, 2, 99], label='Contacto con otro caso'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Toma de muestra de laboratorio'),<br>
&emsp;    gr.Radio(choices=[1, 2, 97, 98, 99], label='Resultado de laboratorio'),<br>
&emsp;    gr.Radio(choices=[1, 2, 98], label='Toma de muestra de antígeno'),<br>
&emsp;    gr.Radio(choices=[1, 2, 97, 98, 99], label='Resultado de antígeno'),<br>
&emsp;    gr.Radio(choices=[1, 2, 97, 98, 99], label='UCI'),<br>
&emsp;    gr.Number(label='Días de Hospitalización',value=0),<br>
&emsp;    gr.Number(label='Días con Síntomas',value=0),<br>
]<br>
<br>
<br>

#### Crear componente de salida para Gradio
outputs = gr.Textbox(label="Predicción")<br>
<br>
<br>

#### Crear la interfaz de Gradio
demo = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)<br>
<br>
<br>

#### Ejecutar la aplicación web
if __name__ == "__main__":<br>
    &emsp;demo.launch(show_api=False,debug=True)<br>

</em>
