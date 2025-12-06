import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from phik import phik_matrix
from sklearn.metrics import r2_score



# 1) Надо загрузить датасет готовый
# 2) построить по самому датасету графики корреляции, распределение признаков (их там немного можно даже от руки) и зависимость тартега от всех признаков. 

# 3) загрузить непосредственно тест, при чем не с кода а в приложении, и развернуть на нем модель,  вывести качества 
# 4) окошко где пользователь может ввести свои признаки
# 5)пайплайн для преобразования признаков (у меня в моделе  есть кодировани и прочее)
# 6) визуализация самых больших весов модели



st.title("Car Price Prediction")


st.title("Графики по тренировочным данным")

DATA_PATH = "df_train_clean.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("Первые строки датасета")
st.dataframe(df.head())

st.subheader("Размер датасета")
st.write(f"Строк: {df.shape[0]}")
st.write(f"Столбцов: {df.shape[1]}")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_cols = cat_cols.drop('name')

st.subheader("Распределения признаков:")

st.subheader("📊 Числовые признаки")

for col in num_cols:
    data = df[col].dropna()

    perc = np.percentile(data, 99)
    data = data[data <= perc]

    fig, ax = plt.subplots()
    ax.hist(data, bins=30)
    ax.set_title(f"Распределение {col}")
    st.pyplot(fig)

st.subheader("🧩 Категориальные признаки")

for col in cat_cols:
    data = df[col].dropna()

    counts = data.value_counts()

    # Ограничим топ-20 категорий (вдруг их много)
    counts = counts.iloc[:10]

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Распределение категорий {col}")
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha='right')
    st.pyplot(fig)


st.header("Корреляция (PHIK) между всеми признаками")

df_corr = df.copy()
num_cols = df_corr.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    perc = np.percentile(df_corr[col].dropna(), 99)
    df_corr.loc[df_corr[col] > perc, col] = perc


phik_corr = df_corr.phik_matrix()
labels = phik_corr.columns
mat = phik_corr.values

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(mat, vmin=0, vmax=1)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

st.pyplot(fig)


st.header("Зависимость признаков и целевой переменной")

target = 'selling_price'
features = [c for c in df.columns if c != target and c !='name']

for col in features:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[col], df[target], alpha=0.6)
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.set_title(f'{col} and {target}')
    
    st.pyplot(fig)


# 2 пункт загрузка датасета и загрузка данных от руки, метод предобработки новых данных и предсказание по ним

st.header("Загрузите датасет и модель посчитает качество!! ")


# Загрузка модели 

# Много кода взято из урока по streamlit Гордея
@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('models/best_ringe/cars_regression.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/best_ringe/cars_regression_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/best_ringe/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('models/best_ringe/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, feature_names, encoder, scaler




model, feature_names, encoder, scaler = load_model()


# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])



# код для преобработки данных
def prepare_features(df, data):
    # небольшая проверка что колонки такие же как в трейне, иначе - вернуть сообщение что не получится
    expected_cols = list(df.columns)
    actual_cols = list(data.columns)
    
    if actual_cols != expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)

        msg = "❌ Невозможно обработать входной датасет: структура не совпадает.\n"
        if missing:
            msg += f"Отсутствуют колонки: {', '.join(missing)}\n"
        if extra:
            msg += f"Лишние колонки: {', '.join(extra)}"
        raise ValueError(msg)
    st.write('Датасет подходит для предсказания')

    # таргет
    y_test = data['selling_price']

    # числовые признаки
    X_test_num = data.select_dtypes(include='number')
    X_test_num = X_test_num.drop(columns=['selling_price'])
    # scaler загружен (надеюсь)
    X_test_num = scaler.transform(X_test_num)


    # добавить 1hot
    data['first_word'] = data['name'].str.split().str[0]
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('name')
    # encoder
    X_test_cat = encoder.transform(data[cat_cols])
    
    
    #Объединяем
    X_final = np.hstack([X_test_num, X_test_cat])

    return X_final, y_test
    

if st.button("Сделать предсказание и посчитать R²"):
    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)

        # Подготовка данных
        try:
            features, y = prepare_features(df, test_data)
        except ValueError as e:
            st.error(str(e))
        else:
            st.write('предобработка завершена!')

            # Предсказание
            predictions = model.predict(features)
            r2 = r2_score(y, predictions)
            
            st.write('Модель отработала, можно оценить качество:')
            st.metric("R² (качество модели)", f"{r2:.3f}")
            # теперь вывести датасет, чтобы пользователь мог посмотреть на предсказания для конкретных примеров
            # сначала добавляем предикт
            test_data["predicted_price"] = predictions

            # корректно меняем порядок
            cols = list(test_data.columns)

            # убрать selling_price и predicted_price из текущих позиций
            cols.remove("selling_price")
            cols.remove("predicted_price")

            # поставить их в конец — в правильном порядке
            cols.append("selling_price")
            cols.append("predicted_price")

            test_data = test_data[cols]

            st.write("Датасет с реальными и предсказанными значениями:")
            st.dataframe(test_data)        
    else:
        st.warning("Сначала загрузите CSV-файл, а потом нажмите кнопку!")

    

st.header("✨ Сделать предсказание вручную")

st.subheader("🔮 Ручное предсказание стоимости")

# выделим категории прямо из train'а
fuel_categories = sorted(df["fuel"].unique())
seller_categories = sorted(df["seller_type"].unique())
trans_categories = sorted(df["transmission"].unique())
owner_categories = sorted(df["owner"].unique())

with st.form("manual_input"):

    st.write("Введите параметры автомобиля, и модель предскажет его стоимость")

    # текст, вводится руками
    name = st.text_input("name (важна только марка!)", value="Maruti")

    # числовые признаки
    year = st.number_input("year", min_value=1900, max_value=2030, step=1, value=2015)
    km_driven = st.number_input("km_driven", min_value=0, step=1000, value=50000)

    mileage = st.number_input("mileage", min_value=0.0, step=0.1, value=20.0)
    engine = st.number_input("engine", min_value=0.0, step=50.0, value=1200.0)
    max_power = st.number_input("max_power", min_value=0.0, step=1.0, value=80.0)
    seats = st.number_input("seats", min_value=1, max_value=15, step=1, value=5)
    max_torque_rpm = st.number_input("max_torque_rpm", min_value=0.0, step=100.0, value=3000.0)
    torque = st.number_input("torque", min_value=0.0, step=1.0, value=100.0)

    # категориальные признаки
    fuel = st.selectbox("fuel", fuel_categories)
    seller_type = st.selectbox("seller_type", seller_categories)
    transmission = st.selectbox("transmission", trans_categories)
    owner = st.selectbox("owner", owner_categories)

    submitted = st.form_submit_button("Получить предсказание")

if submitted:
    try:
        # собираем в DataFrame
        input_dict = {
            "name": [name],
            "year": [year],
            "selling_price" : 500_000,
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats],
            "max_torque_rpm": [max_torque_rpm],
            "torque": [torque],
        }

        input_df = pd.DataFrame(input_dict)

        # готовим как и CSV
        features, _ = prepare_features(df, input_df)

        # предикт
        pred_value = model.predict(features)[0]

        st.success(f"💰 Предсказанная цена: {pred_value:,.0f}")

    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")


st.header('Визуализация важности признаков!')

# удобное хранение коэффициентов
coefs = model.coef_.ravel()
df_coef = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs
})

st.subheader("Важность числовых признаков")
# фильтруем только марки и сортируем
df_brands = df_coef[df_coef["feature"].str.startswith("first_word_")].copy()
df_brands = df_brands.sort_values("coef", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(df_brands["feature"], df_brands["coef"])
plt.xticks(rotation=45, ha='right')
plt.title("Коэффициенты модели для разных марок")
plt.ylabel("Коэффициент")
plt.tight_layout()
st.pyplot(plt.gcf())


# Визуализация числовых признаков
st.subheader("Важность числовых признаков")
numeric_features = [
    "year", "km_driven", "mileage", "engine",
    "max_power", "seats", "max_torque_rpm", "torque"
]

coefs = model.coef_.ravel()
df_coef = pd.DataFrame({"feature": feature_names, "coef": coefs})

df_numeric = df_coef[df_coef["feature"].isin(numeric_features)].copy()
df_numeric = df_numeric.sort_values("coef", ascending=False)


fig_num, ax_num = plt.subplots(figsize=(8, 4))
ax_num.bar(df_numeric["feature"], df_numeric["coef"])
ax_num.set_ylabel("Коэффициент модели")
ax_num.set_title("Numeric features importance")
ax_num.set_xticklabels(df_numeric["feature"], rotation=30, ha="right")
st.pyplot(fig_num)


st.subheader("Важность категориальных признаков")


# Визуализация категориальных признаков
categorical_features = [
    "fuel_Diesel", "fuel_LPG", "fuel_Petrol",
    "seller_type_Individual", "seller_type_Trustmark Dealer",
    "transmission_Manual",
    "owner_Fourth & Above Owner", "owner_Second Owner",
    "owner_Test Drive Car", "owner_Third Owner"
]

df_cat = df_coef[df_coef["feature"].isin(categorical_features)].copy()
df_cat = df_cat.sort_values("coef", ascending=False)

fig_cat, ax_cat = plt.subplots(figsize=(8, 4))
ax_cat.bar(df_cat["feature"], df_cat["coef"])
ax_cat.set_ylabel("Коэффициент модели")
ax_cat.set_title("Categorical features importance")
ax_cat.set_xticklabels(df_cat["feature"], rotation=30, ha="right")
st.pyplot(fig_cat)