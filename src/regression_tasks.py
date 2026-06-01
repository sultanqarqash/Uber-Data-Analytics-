# Step 0: Import all required libraries / الخطوة 0: استيراد جميع المكتبات المطلوبة
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as st

# Add required performance metrics for classification / إضافة مقاييس الأداء المطلوبة للتصنيف
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Step 1: Load fresh CSV file / الخطوة 1: تحميل ملف البيانات الأساسي
df = pd.read_csv("مشروع تخرج اوبر.csv", low_memory=False)

# Drop completely empty rows or rows missing Date/Time/Location to ensure no NaNs in target / حذف الأسطر الفارغة تماماً أو التي لا تحتوي على تاريخ ووقت وموقع لضمان عدم وجود NaN في الهدف
df = df.dropna(subset=['Date', 'Time', 'Pickup Location'])

# Step 2: Change the date and time for the datetime column / الخطوة 2: تغيير التاريخ والوقت لإنشاء عمود التاريخ والوقت المدمج
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Step 3: Sort data by datetime / الخطوة 3: ترتيب البيانات بناءً على التاريخ والوقت
df = df.sort_values('datetime')

# Step 4: Extract the hour and day of the week / الخطوة 4: استخراج الساعة ويوم الأسبوع
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['date_only'] = df['datetime'].dt.date  # Extract date only to link calculations to the actual day / استخراج التاريخ لربط الحساب باليوم الفعلي

# Step 5: Create a weekend column (1 if it's a holiday, 0 if not) / الخطوة 5: إنشاء عمود عطلة نهاية الأسبوع (1 إذا كانت عطلة، 0 إذا كانت يوم عمل)
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Step 6: Clean 'Pickup Location' to remove extra spaces/quotes / الخطوة 6: تنظيف 'موقع الركوب' لإزالة المسافات وعلامات الاقتباس الزائدة
df['Pickup Location'] = df['Pickup Location'].astype(str).str.strip().str.replace('"','')

# Step 7: Calculate previous_hour_rides based on actual order / الخطوة 7: حساب عدد الرحلات في الساعة السابقة بناءً على ترتيب الرحلات الفعلي في نفس اليوم والموقع والساعة
df['previous_hour_rides'] = df.groupby(['Pickup Location', 'date_only', 'hour']).cumcount()

# Step 8: Target Calculation: Calculate actual rides per location, date, and hour / الخطوة 8: حساب المتغير الهدف: حساب عدد الرحلات الفعلي لكل موقع وتاريخ وساعة محددة
df['ride_count'] = df.groupby(['Pickup Location', 'date_only', 'hour']).transform('size')

# Process and clean additional numerical columns to ensure they are free of text or missing values / معالجة وتنظيف الأعمدة الرقمية الإضافية لضمان خلوها من القيم النصية أو الفارغة
df['Ride Distance'] = pd.to_numeric(df['Ride Distance'], errors='coerce').fillna(0)
df['Driver Ratings'] = pd.to_numeric(df['Driver Ratings'], errors='coerce').fillna(4.0)
df['Customer Rating'] = pd.to_numeric(df['Customer Rating'], errors='coerce').fillna(4.0)

# Step 9: Converting category columns to numbers using One-Hot Encoding / الخطوة 9: تحويل الأعمدة الفئوية النصية إلى أرقام باستخدام الترميز الثنائي
categorical_cols = ['Vehicle Type', 'Pickup Location', 'Drop Location', 'Payment Method']
df = pd.get_dummies(df, columns=categorical_cols)

# Step 10: Define Features and Target safely (exclude non-numeric columns and dates) / الخطوة 10: تحديد الميزات والمتغير الهدف بأمان (استبعاد الأعمدة غير الرقمية والهدف والتواريخ)
X = df.select_dtypes(include=[np.number]).drop(columns=['ride_count'], errors='ignore')
y = df['ride_count']

# Final imputation to ensure no NaN values are passed to the model / التبصيم والتحقق النهائي لضمان عدم تمرير أي قيم فارغة بالخطأ للنموذج
X = X.fillna(0)
y = y.fillna(y.median())

# Step 11: Split data into training and testing sets / الخطوة 11: تقسيم البيانات إلى مجموعات تدريب وفحص
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Train Decision Tree Regressor / الخطوة 12: تدريب نموذج شجرة القرار للانجرار
model = DecisionTreeRegressor(random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Step 13: Make predictions / الخطوة 13: إجراء التنبؤات على بيانات الفحص
y_pred = model.predict(X_test)

# Step 14: Evaluate model using Mean Absolute Error (MAE) / الخطوة 14: تقييم النموذج باستخدام متوسط الخطأ المطلق
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# -------------------------------------------------------------------------
# New Additions: Performance Metrics (Accuracy, Confusion Matrix, & Report)
# إضافات جديدة: مقاييس الأداء (الدقة، مصفوفة الارتباك، والتقرير التفصيلي)
# -------------------------------------------------------------------------

# Round predicted values to nearest integer to fit classification metrics / نقوم بتقريب القيم المتنبأ بها لأقرب رقم صحيح لملائمة مقاييس التصنيف
y_test_discrete = y_test.round().astype(int)
y_pred_discrete = np.clip(y_pred.round(), 1, 5).astype(int)

# 1. Calculate and print Accuracy Score / حساب وطباعة مقياس نسبة الدقة
accuracy = accuracy_score(y_test_discrete, y_pred_discrete)
print(f"Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 2. Calculate and print Confusion Matrix / حساب وطباعة مصفوفة الارتباك
cm = confusion_matrix(y_test_discrete, y_pred_discrete)
print("\nConfusion Matrix:")
print(cm)

# 3. Print detailed Classification Report (Precision, Recall, F1-Score) / طباعة تقرير التصنيف التفصيلي (الدقة، الاستدعاء، ومقياس F1)
cr = classification_report(y_test_discrete, y_pred_discrete, zero_division=0)
print("\nClassification Report:")
print(cr)

# 4. Plot Confusion Matrix Heatmap for the report / رسم مصفوفة الارتباك كخريطة حرارية ملونة للتقرير
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test_discrete),
            yticklabels=np.unique(y_test_discrete))
plt.title('Confusion Matrix Heatmap - Ride Count')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# -------------------------------------------------------------------------

# Step 15: Plot Actual vs Predicted Ride Count (Line Plot) / الخطوة 15: رسم مخطط مقارنة القيم الحقيقية ضد المتوقعة (الرسمة الأولى: الخطوط)
plt.figure(figsize=(12,5))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('Decision Tree Regression: Actual vs Predicted Ride Count')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png') # Save plot image / حفظ الرسمة كصورة
plt.show()

# Step 16: Plot the Decision Tree Structure (Tree diagram for the report) / الخطوة 16: رسم بنية شجرة القرار (الرسمة الثانية: الشجرة المتفرعة للتقرير)
plt.figure(figsize=(25,12))
st.plot_tree(model, max_depth=3, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure (First 3 Levels)")
plt.tight_layout()
plt.savefig('decision_tree_structure.png')
plt.show()