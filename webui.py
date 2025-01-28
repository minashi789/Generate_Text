import numpy as np
import tensorflow as tf
import re
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import nltk
from langdetect import detect
import matplotlib.pyplot as plt
import logging
import time

# Убедитесь, что у вас установлена версия TensorFlow 2.x
assert tf.__version__.startswith('2.'), "Требуется TensorFlow версии 2.x"

# Скачиваем необходимые ресурсы для nltk
nltk.download('punkt')

# Настройка логирования в файл
log_file = "training.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функция предварительной обработки текста с поддержкой различных языков
def preprocess_text(text):
    try:
        # Определение языка текста с использованием langdetect
        detected_lang = detect(text)
        logging.info(f"Detected language: {detected_lang}")
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Заменяем специальные символы на стандартизированные
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Замена кавычек
        text = re.sub(r'[\u2013\u2014]', '-', text)  # Замена тире
        
        # Удаляем нежелательные символы, оставляя основные знаки препинания
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Учитываем многоточие и сокращения
        text = re.sub(r'\bт\.д\.\b', 'т.д.', text)
        text = re.sub(r'\bи т\.п\.\b', 'и т.п.', text)
        
        # Заменяем множественные пробелы на одинарные
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Убираем пробелы перед знаками препинания
        text = re.sub(r'\s([.,!?;:])', r'\1', text)
        
        return text
    except Exception as e:
        logging.error(f"Error in preprocess_text: {e}")
        return ""

# Загрузка и предобработка текста из файла
def load_and_preprocess_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return preprocess_text(text)
    except Exception as e:
        logging.error(f"Error loading and preprocessing text from {file_path}: {e}")
        return ""

# Быстрое разбиение текста на предложения с помощью регулярных выражений
def split_into_sentences(text):
    try:
        # Используем регулярное выражение для разделения на предложения
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
        return sentences
    except Exception as e:
        logging.error(f"Error splitting into sentences: {e}")
        return []

# Создание последовательностей и словарей с параллельной обработкой
def create_sequences(text, seq_length):
    try:
        sentences = split_into_sentences(text)
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(sentences)
        total_words = len(tokenizer.word_index) + 1
        
        def process_sentence(sentence):
            token_list = tokenizer.texts_to_sequences([sentence])[0]
            input_sequences = []
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
            return input_sequences
        
        input_sequences = Parallel(n_jobs=-1)(delayed(process_sentence)(sentence) for sentence in sentences)
        input_sequences = [seq for sublist in input_sequences for seq in sublist]  # Flatten list of lists
        
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        
        X, y = input_sequences[:,:-1], input_sequences[:,-1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        return X, y, tokenizer, max_sequence_len, total_words
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        return None, None, None, None, None

# Построение модели с параметризацией output_dim и добавлением слоя LayerNormalization и Attention
def build_model(vocab_size, max_sequence_len, embedding_dim=128):
    inputs = tf.keras.Input(shape=(max_sequence_len-1,))
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_len-1)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Функция для генерации текста с использованием beam search
def beam_search_generate_text(seed_text, model, tokenizer, max_sequence_len, num_words, beam_width=3, max_sentence_length=50):
    try:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        generated_text = seed_text
        sentence_length = 0
        
        beam = [(token_list, generated_text, 0.0)]
        
        for _ in range(num_words):
            new_beam = []
            for seq, text, score in beam:
                predicted = model.predict(seq, verbose=0)[0]
                top_indices = np.argsort(predicted)[-beam_width:]
                top_probs = predicted[top_indices]
                
                for index, prob in zip(top_indices, top_probs):
                    new_token_list = np.append(seq[:, 1:], [[index]], axis=1)
                    new_text = text + " " + tokenizer.index_word.get(index, "")
                    new_score = score - np.log(prob)  # Минимизируем отрицательную логарифмическую вероятность
                    new_beam.append((new_token_list, new_text, new_score))
            
            beam = sorted(new_beam, key=lambda x: x[2])[:beam_width]
            
            if beam:
                best_seq, best_text, _ = beam[0]
                generated_text = best_text
                token_list = best_seq
                sentence_length += 1
                
                # Ограничение длины предложения
                if sentence_length >= max_sentence_length and best_text.split()[-1] in ['.', '?', '!']:
                    break
        
        # Постобработка текста: удаляем лишние пробелы перед знаками препинания
        generated_text = re.sub(r'\s+([.,!?;:])', r'\1', generated_text)
        return generated_text
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return ""

# Экспоненциальный спад для LearningRateScheduler
def lr_scheduler(epoch, lr):
    initial_lr = 0.001  # Начальное значение lr
    decay_rate = 0.95  # Скорость спада
    return initial_lr * (decay_rate ** epoch)

# Функция для построения графиков метрик
def plot_metrics(history):
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig("training_metrics.png")
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting metrics: {e}")

# Инструкция по запуску TensorBoard
"""
Для использования TensorBoard запустите следующую команду в терминале:
tensorboard --logdir=./logs
"""

# Установка необходимых библиотек
# Убедитесь, что все необходимые библиотеки установлены
# pip install tensorflow nltk langdetect matplotlib scikit-learn joblib

# Загрузка и предобработка текста
start_time = time.time()
text = load_and_preprocess_text('Granatovyj_braslet.txt')
logging.info(f"Text preprocessing completed in {time.time() - start_time:.2f} секунд")

if text:
    # Создание последовательностей и словарей
    start_time = time.time()
    seq_length = 10
    X, y, tokenizer, max_sequence_len, vocab_size = create_sequences(text, seq_length)
    logging.info(f"Sequence creation completed in {time.time() - start_time:.2f} секунд")
    
    if X is not None and y is not None and tokenizer is not None and max_sequence_len is not None and vocab_size is not None:
        # Разделение набора данных на тренировочный и валидационный с использованием train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Подготовка данных и построение модели
        model = build_model(vocab_size, max_sequence_len, embedding_dim=128)
        
        # Добавляем колбэк ModelCheckpoint для сохранения только весов после каждой эпохи с лучшими параметрами
        checkpoint_path = "training/cp.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1)
        
        # Добавляем Learning Rate Scheduler с экспоненциальным спадом
        lr_callback = LearningRateScheduler(lr_scheduler)
        
        # Добавляем EarlyStopping для автоматической остановки обучения при отсутствии улучшений
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        
        # Добавляем TensorBoard для визуализации метрик
        tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)
        
        callbacks = [cp_callback, lr_callback, early_stopping, tensorboard_callback]
        
        # Преобразование данных в tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
        
        # Обучение модели с использованием коллбэков
        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=100,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        logging.info(f"Model training completed in {time.time() - start_time:.2f} секунд")
        
        # Построение графиков метрик
        start_time = time.time()
        plot_metrics(history)
        logging.info(f"Metrics plotting completed in {time.time() - start_time:.2f} секунд")
        
        # Генерация текста с использованием beam search
        seed_text = "И от лености или со скуки"
        start_time = time.time()
        generated_text = beam_search_generate_text(seed_text, model, tokenizer, max_sequence_len, num_words=50, beam_width=3, max_sentence_length=50)
        logging.info(f"Text generation completed in {time.time() - start_time:.2f} секунд")
        print(generated_text)
        
        # Сохранение полной модели в новом формате .keras
        start_time = time.time()
        model.save("training/final_model.keras")
        logging.info(f"Model saving completed in {time.time() - start_time:.2f} секунд")
    else:
        logging.error("Failed to create sequences or tokenizer.")
else:
    logging.error("Failed to load and preprocess text.")