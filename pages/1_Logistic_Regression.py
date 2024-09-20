import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.lines import Line2D


# задаем параметры отображения страницы, иконку и тектовое наполнение
st.set_page_config(page_title='Logistic Regression', page_icon='👻')
st.title(
    'Использование Logistic Regression'
    )

st.markdown("""
### Logistic Regression
Пользователь в соответствующее поле должен загрузить подготовленный датасет,
на котором он хочет обучить модель логистической регрессии. Датасет не должен
содержать в себе никаких типов данных, за исключением численных! В явном виде
должен присутствовать один целевой (target) столбец и нерегламентированное число
фичей (features). 
""")

upload_file = st.file_uploader(
    label='Загрузите подготовленный для обучения датасет (CSV)',
    type='csv'
    )

# кэшированная функция загрузки датасета пользователем
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

if upload_file is not None:
    # получаем загруженный юзером датасет
    df = load_data(uploaded_file=upload_file)

    # выводим первые 5 строчек, чтобы удостоверить юзера, что датасет загружен верно
    st.write('### Первые 5 строчек загруженного датасета')
    st.write(df.head(5))

    # определяем класс логистической регрессии и считаем дальше
    class LogReg:

        def __init__(self, learning_rate, n_epochs):
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.coef_ = None
            self.intercept_ = None


        def sigmoida(self, x):
            return 1 / (1 + np.exp(-x))


        def fit(self, X, y):
            # переведем полученные значения в numpy для корректных преобразований
            X = np.array(X)
            y = np.array(y)

            # инициализируем веса и смещение в зависимости от введенных данных
            n_samples, n_features = X.shape
            self.coef_ = np.random.uniform(low=0.0, high=1.0, size=n_features)
            self.intercept_ = 0

            # запустим цикл, равный числе эпох
            for _ in range(self.n_epochs):
                # предсказанное значение по первым инициализированным весам и смещению
                y_pred = self.sigmoida(np.dot(X, self.coef_) + self.intercept_)

                # посчитаем градиенты для w0 и w
                dw0_grad = -(y - y_pred)
                dw_grad = -X * (y - y_pred).reshape(-1, 1)

                # обновим веса
                self.intercept_ = self.intercept_ - self.learning_rate * dw0_grad.mean()
                self.coef_ = self.coef_ - self.learning_rate * dw_grad.mean(axis=0)


        def predict_prob(self, X):
            X = np.array(X)
            y_pred = self.sigmoida(np.dot(X, self.coef_) + self.intercept_)
            return y_pred

        def predict(self, X):
            X = np.array(X)
            class_pred = [0 if y <= 0.5 else 1 for y in self.predict_prob(X)]
            return class_pred

    # запрашиваем у пользователя, что будет таргетом
    name_target_column = st.selectbox("Выберите целевую переменную:", df.columns)
    target_column = df[name_target_column]
    train_columns = df.drop(name_target_column, axis=1)

    # выводим пользователю, чтобы удостоверился он, что все чики-пуки
    st.write('### Первые 5 строчек трейнового датасета без таргета')
    st.write(train_columns.head())

    # запрашиваем ввод learning rate
    st.write('### Введите желаемый learning rate (по умолчанию равен 0.01)')
    user_lr = st.number_input(
        'Введите значение learning rate',
        min_value=0.0001,
        max_value=0.2,
        value=0.01,
        step=0.05
        )
    
    # запрашиваем ввод n_epochs
    st.write('### Введите желаемое время обучение (кол-во эпох) (по умолчанию - 101)')
    user_epochs = st.number_input(
        'Введите количество эпох',
        min_value=1,
        max_value=1001,
        value=101,
        step=50
        )
    if st.button('Обучить модель и получить веса и смещения!'):
        # создаем экземпляр класса
        new_log_reg = LogReg(learning_rate=user_lr, n_epochs=user_epochs)
        new_log_reg.fit(X=train_columns, y=target_column)

        # создаем словарь с ответами для пользователя
        answ = dict(zip(train_columns.columns, new_log_reg.coef_))
        intercept = new_log_reg.intercept_

        # выводим пользователю
        st.write("**Коэффициенты модели:**")
        st.table(pd.DataFrame({
            'Фича': answ.keys(),
            'Вес (Коэффициент)': answ.values()
        }))
        st.write(f"**Свободный член (intercept):** {intercept}")
    else:
        st.stop()
    # if st.button('Построить графики, хоть какие-нибудь 😺'):
        
    #     fig, axes = plt.subplots(figsize=(12, 6))
        
    #     X_array = train_columns.values

    #     x_ax = X_array[:, 0]
    #     y_ax = X_array[:, 1]

    #     pred_values = new_log_reg.predict(train_columns)

    #     colors = ['red' if y_pred > 0.5 else 'blue' for y_pred in pred_values]

    #     scatter_pred = axes.scatter(x=x_ax, y=y_ax, s=pred_values*20, c=colors, alpha=0.9) # тут умножил на 20 ради масштабирования точек графика

    #     # добавление разделяющей прямой
    #     x_min = x_ax.min() - 1
    #     x_max = x_ax.max() + 1
    #     x_values = np.linspace(x_min, x_max, 100)
    #     # Формула разделяющей прямой: self.coef_[0] * x + self.coef_[1] * y + self.intercept_ = 0
    #     # Решаем относительно y: y = (-self.coef_[0] * x - self.intercept_) / self.coef_[1]
    #     y_values = (-new_log_reg.coef_[0] * x_values - new_log_reg.intercept_) / new_log_reg.coef_[1]

    #     plt.plot(x_values, y_values, color='green', linestyle='--', linewidth=2)

    #     # кастомное добавление легенды
    #     legend_elements = [
    #         Line2D([0], [0], marker='o', color='w', label='pred > 0.5 - считаем 1',
    #             markerfacecolor='red', markersize=10),
    #         Line2D([0], [0], marker='o', color='w', label='pred ≤ 0.5 - считаем 0',
    #             markerfacecolor='blue', markersize=10),
    #         Line2D([0], [0], color='green', lw=2, linestyle='--',
    #                                 label='Разделяющая прямая'),
    #     ]

    #     plt.legend(handles=legend_elements, title='Предсказания')

    #     # Добавление заголовков и меток осей
    #     plt.title('Предсказания логистической регрессии')
    #     plt.xlabel('Фича 1 (CCAvg)')
    #     plt.ylabel('Фича 2 (Income)')

    #     # Отображение сетки для лучшей читаемости
    #     plt.grid(True, linestyle='--', alpha=0.5)
        
    #     plt.tight_layout()
    #     st.pyplot(fig)

    # else:
    #     st.stop()
else:
    st.stop()
