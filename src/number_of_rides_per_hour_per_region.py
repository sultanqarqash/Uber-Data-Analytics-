# =========================================================================
# Step 0: Import All Required Libraries
# =========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report

# استدعاء مكتبات التعلم العميق (TensorFlow & Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# Step 1: Load and Clean Dataset
# =========================================================================
print("جاري تحميل وقراءة ملف البيانات...")
# ملاحظة: تأكد من وجود الملف في نفس مسار تشغيل الكود
df = pd.read_csv("مشروع تخرج اوبر.csv", low_memory=False)

# تنظيف وتصفية الأسطر لضمان خلوها من القيم الفارغة في الأعمدة الأساسية
df = df.dropna(subset=['Date', 'Time', 'Pickup Location', 'Drop Location'])

# =========================================================================
# Step 2: Extract Time & Date Features (الميزات الزمنية)
# =========================================================================
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('datetime')

# 1. Hour of the day
df['hour'] = df['datetime'].dt.hour

# 2. Day of the week
df['day_of_week'] = df['datetime'].dt.dayofweek

# 9. Weekend / Holiday flag (1 لنهاية الأسبوع، 0 لباقي الأيام)
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

df['date_only'] = df['datetime'].dt.date  # متغير وسيط لربط الحسابات باليوم الفعلي

# =========================================================================
# Step 3: Clean Text Columns & Handle Strings
# =========================================================================
df['Pickup Location'] = df['Pickup Location'].astype(str).str.strip().str.replace('"','')
df['Drop Location'] = df['Drop Location'].astype(str).str.strip().str.replace('"','')
df['Vehicle Type'] = df['Vehicle Type'].astype(str).str.strip()
df['Payment Method'] = df['Payment Method'].astype(str).str.strip()

# =========================================================================
# Step 4: Create Target and Lag Features
# =========================================================================
# 10. Previous hour rides (lag feature)
df['previous_hour_rides'] = df.groupby(['Pickup Location', 'date_only', 'hour']).cumcount()

# TARGET: Number of rides per hour per region (المتغير المستهدف)
df['ride_count'] = df.groupby(['Pickup Location', 'date_only', 'hour']).transform('size')

# =========================================================================
# Step 5: Process Continuous Numeric Features
# =========================================================================
# 5. Ride Distance
df['Ride Distance'] = pd.to_numeric(df['Ride Distance'], errors='coerce').fillna(0)

# 6. Avg VTAT (وقت وصول السائق)
df['Avg VTAT'] = pd.to_numeric(df['Avg VTAT'], errors='coerce').fillna(df['Avg VTAT'].median())

# 7. Avg CTAT (مدة الرحلة الفعالة)
df['Avg CTAT'] = pd.to_numeric(df['Avg CTAT'], errors='coerce').fillna(df['Avg CTAT'].median())

# =========================================================================
# Step 6: One-Hot Encoding for Categorical Columns
# =========================================================================
# 3. Region/Area (Pickup & Drop Location) & 4. Vehicle Type & 8. Payment Method
categorical_cols = ['Pickup Location', 'Drop Location', 'Vehicle Type', 'Payment Method']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =========================================================================
# Step 7: Define Features (X) and Target (y)
# =========================================================================
# تجميع الـ 10 ميزات المطلوبة بعد تحويل الأعمدة النصية إلى رقمية
base_features = ['hour', 'day_of_week', 'weekend', 'Ride Distance', 'Avg VTAT', 'Avg CTAT', 'previous_hour_rides']
encoded_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_cols)]
X_cols = base_features + encoded_cols

X = df_encoded[X_cols].astype(np.float32)
y = df_encoded['ride_count'].values.astype(np.float32)

# فحص نهائي لضمان خلو مصفوفة المدخلات من أي قيم مفقودة
X = X.fillna(0)

# =========================================================================
# Step 8: Train-Test Split (تقسيم البيانات)
# =========================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================================
# Step 9: Feature Scaling (تطبيع وموازنة البيانات)
# =========================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================================
# Step 10: Build Neural Network Architecture
# =========================================================================
print("\nجاري بناء هيكلية الشبكة العصبية الاصطناعية...")
model = Sequential([
    # الطبقة الأولى المدخلة (128 نيورون)
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2), # لحماية النموذج من الحفظ الأعمى والـ Overfitting

    # الطبقة المخفية الثانية (64 نيورون)
    Dense(64, activation='relu'),
    Dropout(0.2),

    # الطبقة المخفية الثالثة (32 نيورون)
    Dense(32, activation='relu'),

    # طبقة المخرجات (Linear لأنها مهمة Regression للتنبؤ بالقيم العددية)
    Dense(1, activation='linear')
])

# عرض ملخص هيكلية الشبكة
model.summary()

# =========================================================================
# Step 11: Compile and Train the Model
# =========================================================================
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# تفعيل التوقف المبكر في حال استقرار الأداء لمنع الإفراط في التدريب
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nبدء تدريب الشبكة العصبية الآن...")
history = model.fit(X_train_scaled, y_train,
                    epochs=15,
                    batch_size=256,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# =========================================================================
# Step 12: Performance Evaluation & Metrics Generation
# =========================================================================
print("\n" + "="*50)
print("إحصائيات ومقاييس الأداء النهائية للشبكة العصبية")
print("="*50)

# 1. حساب الـ MAE الأساسي للنموذج الرقمي
y_pred = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred)
print(f"1. Mean Absolute Error (MAE): {mae:.4f}")

# 2. تقريب التوقعات لأقرب أرقام صحيحة لاستخراج مقاييس التصنيف (Accuracy & Matrix)
y_test_discrete = np.round(y_test).astype(int)
y_pred_discrete = np.clip(np.round(y_pred), 1, 5).astype(int)

# حساب الدقة الشاملة Accuracy Score
accuracy = accuracy_score(y_test_discrete, y_pred_discrete)
print(f"2. Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# حساب مصفوفة الارتباك Confusion Matrix
cm = confusion_matrix(y_test_discrete, y_pred_discrete)
print("\n3. Confusion Matrix:")
print(cm)

# طباعة التقرير التصنيفي الشامل
print("\n4. Comprehensive Performance Report:")
print(classification_report(y_test_discrete, y_pred_discrete, zero_division=0))

# =========================================================================
# Step 13: Data Visualization (إنتاج الرسومات والمنحنيات للتقرير)
# =========================================================================
print("\nجاري رسم وحفظ المخططات البيانية...")

# 1. رسم منحنيات التعلم (Learning Curves)
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train MAE / Loss')
plt.plot(history.history['val_loss'], label='Validation MAE / Loss')
plt.title('Neural Network Learning Curves (MAE Loss)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nn_ride_count_learning_curves.png')
plt.show()

# 2. رسم مصفوفة الارتباك الحرارية (Confusion Matrix Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=np.unique(y_test_discrete),
            yticklabels=np.unique(y_test_discrete))
plt.title('Neural Network Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png')
plt.show()

# 3. رسم مقارنة بين القيم الحقيقية والمتوقعة (أول 100 عينة) -
plt.figure(figsize=(12, 6))
plt.plot(y_test_discrete[:100], label='Actual Ride Count', color='blue', marker='o', alpha=0.7)
plt.plot(y_pred_discrete[:100], label='Predicted Ride Count', color='orange', linestyle='--', marker='x', alpha=0.8)
plt.title('Neural Network: Actual vs Predicted Ride Count (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Ride Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nn_actual_vs_predicted.png')
plt.show()