# Step 0: Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# استدعاء مكتبات التعلم العميق
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load fresh CSV file
print("جاري تحميل البيانات وتجهيزها...")
df = pd.read_csv("مشروع تخرج اوبر.csv", low_memory=False)

# حذف الأسطر التي لا تحتوي على تاريخ أو حالة للحفاظ على نظافة البيانات
df = df.dropna(subset=['Date', 'Time', 'Booking Status'])

# Step 2: Preprocess Date and Time
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Step 3: Create the Classification Target
# 1 إذا اكتملت الرحلة، 0 إذا ألغيت من السائق أو الزبون أو لم يجدوا سائقاً
df['target'] = df['Booking Status'].apply(lambda x: 1 if x == 'Completed' else 0)

# Step 4: Clean numeric columns
df['Ride Distance'] = pd.to_numeric(df['Ride Distance'], errors='coerce').fillna(0)
df['Driver Ratings'] = pd.to_numeric(df['Driver Ratings'], errors='coerce').fillna(4.0)
df['Customer Rating'] = pd.to_numeric(df['Customer Rating'], errors='coerce').fillna(4.0)
df['Booking Value '] = pd.to_numeric(df['Booking Value '], errors='coerce').fillna(0)

# Step 5: Convert categorical columns into dummy variables
categorical_cols = ['Vehicle Type', 'Pickup Location', 'Payment Method']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 6: Define Features (X) and Target (y)
# اختيار الميزات الأساسية بالإضافة للأعمدة الناتجة عن الـ get_dummies
base_features = ['hour', 'day_of_week', 'weekend', 'Ride Distance', 'Driver Ratings', 'Customer Rating', 'Booking Value ']
encoded_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_cols)]
X_cols = base_features + encoded_cols

X = df_encoded[X_cols].astype(np.float32)
y = df_encoded['target'].values

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Data Scaling (مهم جداً جداً لنجاح واستقرار الشبكات العصبية)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Build the Neural Network Architecture
print("جاري بناء هيكلية الشبكة العصبية...")
model = Sequential([
    # الطبقة الأولى المدخلة مع 64 نيورون
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2), # لحماية النموذج من Overfitting

    # الطبقة المخفية الثانية
    Dense(32, activation='relu'),
    Dropout(0.2),

    # الطبقة المخرجة (sigmoid لأنها تصنيف ثنائي: 0 أو 1)
    Dense(1, activation='sigmoid')
])

# تفاصيل بناء النموذج
model.summary()

# Step 10: Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# إضافة خاصية التوقف المبكر في حال ثبت الأداء لعدم تضييع الوقت
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 11: Train the Neural Network
print("بدء تدريب الشبكة العصبية...")
history = model.fit(X_train_scaled, y_train,
                    epochs=15,
                    batch_size=256,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# Step 12: Evaluate model on test data
print("\n--- تقييم النموذج على بيانات الفحص ---")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten() # تحويل الاحتمالات إلى 0 أو 1 بناءً على العتبة 0.5

# حساب الـ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# حساب الـ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# طباعة تقرير التصنيف (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------------------------
# الرسومات البيانية    (Plots)
# -------------------------------------------------------------------------

# 1. رسم منحنى الدقة أثناء التدريب (Training & Validation Accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 2. رسم منحنى الخسارة أثناء التدريب (Training & Validation Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('nn_learning_curves.png')
plt.show()

# 3. رسم مصفوفة الارتباك بشكل احترافي (Confusion Matrix Heatmap)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Cancelled/Incomplete', 'Completed'],
            yticklabels=['Cancelled/Incomplete', 'Completed'])
plt.title('NN Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png')
plt.show()