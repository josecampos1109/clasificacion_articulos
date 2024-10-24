from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)  # Corregido: __name__

# Datos de ejemplo
ejemplos_titulos = [
    "El clima es cálido hoy",
    "La economía está en crecimiento",
    "El fútbol es el deporte más popular",
    "Las nuevas tecnologías están cambiando el mundo",
    "La salud mental es importante"
]
ejemplos_categorias = [
    "Clima",
    "Economía",
    "Deporte",
    "Tecnología",
    "Salud"
]

# Preprocesamiento del texto y etiquetas
def preprocesar_datos(titulos, categorias):
    # Convertir las etiquetas a números
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categorias)

    # Tokenización del texto
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(titulos)
    sequences = tokenizer.texts_to_sequences(titulos)

    # Cambiar la longitud máxima a 10 para que coincida con el modelo
    X = pad_sequences(sequences, maxlen=10)

    return X, labels, tokenizer, label_encoder

# Preprocesar los datos de ejemplo
X, labels, tokenizer, label_encoder = preprocesar_datos(ejemplos_titulos, ejemplos_categorias)

# Crear y compilar el modelo de Keras
def crear_modelo():
    model = keras.Sequential([
        layers.Embedding(input_dim=20000, output_dim=128, input_length=10),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(np.unique(labels)), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = crear_modelo()

# Entrenar el modelo con los datos de ejemplo
model.fit(X, labels, epochs=10)

# Función para clasificar el texto del usuario
def clasificar_texto(texto):
    seq = tokenizer.texts_to_sequences([texto])
    padded = pad_sequences(seq, maxlen=10)
    prediccion = model.predict(padded)
    index_categoria = np.argmax(prediccion)
    categoria = label_encoder.inverse_transform([index_categoria])
    return {'categoria': categoria[0]}

@app.route('/clasificar', methods=['POST'])
def clasificar():
    try:
        datos = request.get_json()
        texto_usuario = datos.get('texto')

        # Validación de entrada
        if not texto_usuario:
            return jsonify({'error': 'No se proporcionó texto para clasificar'}), 400
        if not isinstance(texto_usuario, str):
            return jsonify({'error': 'El texto debe ser una cadena de caracteres'}), 400
        if len(texto_usuario.strip()) == 0:
            return jsonify({'error': 'El texto no puede estar vacío'}), 400
        if len(texto_usuario) > 100:  # Limitar longitud del texto
            return jsonify({'error': 'El texto no puede exceder los 100 caracteres'}), 400

        resultado = clasificar_texto(texto_usuario)
        return jsonify({
            'titulo': texto_usuario,
            'categoria': resultado['categoria']
        })
    except Exception as e:
        print(f"Error en la solicitud POST: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  # Agregado para ejecutar la aplicación
