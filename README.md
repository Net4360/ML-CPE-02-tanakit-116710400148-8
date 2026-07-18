ML-LAB-02: Data Preprocessing
นำเข้าไลบรารีที่จำเป็น ก่อนเริ่มทำLab
pandas (pd): ใช้สำหรับจัดการและวิเคราะห์ข้อมูลในรูปแบบตาราง (DataFrame)
numpy (np): ใช้สำหรับคำนวณทางคณิตศาสตร์และจัดการกับค่าสูญหาย (NaN)
matplotlib.pyplot (plt) และ seaborn (sns): ใช้สำหรับสร้างกราฟและภาพจำลองข้อมูล (Data Visualization)
LabelEncoder: ใช้สำหรับแปลงข้อมูลประเภทข้อความ (Categorical) ให้เป็นตัวเลข

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LAB 1: Dataset Exploration (การสำรวจชุดข้อมูล)

df = pd.read_csv('student_dataset_v3.csv')
pd.read_csv(...): ทำการโหลดไฟล์ข้อมูล CSV เข้ามาเก็บไว้ในตัวแปร df (DataFrame)

print(df.shape)
print(df.dtypes)
print(df.describe(include='all'))
shape: แสดงขนาดของข้อมูล (จำนวนแถว, จำนวนคอลัมน์)
dtypes: ตรวจสอบประเภทของข้อมูลในแต่ละคอลัมน์ (เช่น int, float, object)
describe(include='all'): แสดงค่าสถิติเชิงพรรณนาพื้นฐาน เช่น ค่าเฉลี่ย (mean), ค่าเบี่ยงเบนมาตรฐาน (std), ค่าต่ำสุด/สูงสุด รวมถึงจำนวนค่าที่ไม่ซ้ำกันของข้อมูลแบบข้อความ

print(df.isnull().sum())
print(df.duplicated().sum())
print(df['Passed'].value_counts())
isnull().sum(): นับจำนวนค่าที่สูญหาย (Missing Values) ในแต่ละคอลัมน์
duplicated().sum(): นับจำนวนแถวที่มีข้อมูลซ้ำซ้อนกันทุกคอลัมน์ (Duplicate Records)
value_counts(): นับจำนวนข้อมูลในคอลัมน์เป้าหมาย ('Passed') เพื่อดูการกระจายตัวของคลาส (Class Distribution) ว่าสมดุลหรือไม่

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LAB 2: Data Visualization (การแสดงภาพข้อมูล)

df_viz = df.copy()
df_viz['Study_Hours'] = pd.to_numeric(df_viz['Study_Hours'], errors='coerce')
สร้างตัวแปร df_viz เพื่อไม่ให้กระทบข้อมูลหลัก และบังคับแปลงคอลัมน์ 'Study_Hours' ให้เป็นตัวเลข (to_numeric) หากเจอข้อความที่แปลงไม่ได้ให้เปลี่ยนเป็นค่า NaN (errors='coerce')

plt.hist(df_viz['Exam_Score'].dropna(), bins=10, edgecolor='black', color='#1f77b4')
plt.hist(...): สร้างกราฟ Histogram ของคะแนนสอบ
dropna(): ตัดค่าว่างทิ้งก่อนพล็อตเพื่อไม่ให้เกิด Error
bins=10: แบ่งช่วงของข้อมูลออกเป็น 10 แท่ง
edgecolor='black': ตีเส้นขอบกราฟแท่งสีดำให้ดูง่ายขึ้น

numeric_cols = df_viz[['Age', 'Study_Hours', 'Exam_Score']]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
numeric_cols.corr(): คำนวณค่าสหสัมพันธ์ (Correlation) ระหว่างตัวแปรเชิงปริมาณ
sns.heatmap(...): สร้าง Heatmap เพื่อแสดงระดับความสัมพันธ์
annot=True: ให้แสดงตัวเลขค่า Correlation บนกราฟ
cmap='coolwarm': ใช้โทนสีเย็น-ร้อนเพื่อแยกความแตกต่าง (สีแดง=สัมพันธ์เชิงบวก, สีน้ำเงิน=สัมพันธ์เชิงลบ)
fmt=".2f": แสดงตัวเลขทศนิยม 2 ตำแหน่ง

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Part 3: Data Cleaning (การทำความสะอาดข้อมูล)

df_clean = df.copy()
df_clean['Study_Hours'] = pd.to_numeric(df_clean['Study_Hours'], errors='coerce')
Data Type Conversion: แปลงประเภทข้อมูลให้ถูกต้องสำหรับการคำนวณ

df_clean['Gender'] = df_clean['Gender'].replace({'female': 'Female', 'F': 'Female', 'M': 'Male'})
df_clean.loc[(df_clean['Age'] <= 0) | (df_clean['Age'] > 60), 'Age'] = np.nan
Incorrect Data Correction:
บรรทัดแรก: แก้ไขความไม่สอดคล้องของการพิมพ์ (Inconsistent Data) ในคอลัมน์เพศ ให้เป็นรูปแบบมาตรฐานเดียวกัน
บรรทัดที่สอง: จัดการค่าผิดปกติ (Outliers) เช่น อายุที่ติดลบ หรืออายุเกิน 60 ปี โดยสั่งให้แทนค่าเหล่านั้นด้วย NaN (ค่าว่าง) ไปก่อน

df_clean = df_clean.drop_duplicates().reset_index(drop=True)
Duplicate Removal: คำสั่ง .drop_duplicates() ลบแถวที่ซ้ำซ้อนออก และ .reset_index(drop=True) ทำการจัดเรียงเลข Index ใหม่ให้ต่อเนื่องกัน

mean_score = df_clean['Exam_Score'].mean()
median_score = df_clean['Exam_Score'].median()
Compare Mean/Median: คำนวณหาค่าเฉลี่ยและค่ามัธยฐานเพื่อเปรียบเทียบดูว่าข้อมูลเบ้หรือไม่

df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Study_Hours'] = df_clean['Study_Hours'].fillna(df_clean['Study_Hours'].median())
df_clean['Exam_Score'] = df_clean['Exam_Score'].fillna(df_clean['Exam_Score'].median())
Missing Value Handling: เติมค่าว่าง (.fillna()) ด้วยค่ามัธยฐาน (.median()) การใช้ค่ามัธยฐานจะช่วยลดผลกระทบจาก Outliers ได้ดีกว่าการใช้ค่าเฉลี่ย

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Part 4: Feature Engineering (การจัดการและสร้างคุณลักษณะ)

le = LabelEncoder()
df_clean['Passed_Encoded'] = le.fit_transform(df_clean['Passed'])
Label Encoding: ใช้คำสั่ง .fit_transform() แปลงคอลัมน์เป้าหมาย 'Passed' ที่มีค่า (Yes/No) ให้กลายเป็นตัวเลข (1/0)

df_final = pd.get_dummies(df_clean, columns=['Major', 'Gender'], drop_first=True)
One-Hot Encoding: ใช้ .get_dummies() แตกคอลัมน์ข้อมูลหมวดหมู่ (Categorical) เช่น 'Major' และ 'Gender' ให้กลายเป็นคอลัมน์ย่อยที่เป็นค่า 0 และ 1 (True/False)
drop_first=True: ใช้สำหรับตัดคอลัมน์แรกทิ้งเพื่อป้องกันปัญหา Dummy Variable Trap (ความซ้ำซ้อนของตัวแปรที่โมเดลสามารถอนุมานค่าได้เอง)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
