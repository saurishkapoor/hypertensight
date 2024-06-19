import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO
from fpdf import FPDF
from datetime import date, datetime

# Function to preprocess the image (green channel + CLAHE)
def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    green_channel = image_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(green_channel)
    processed_image = cv2.merge([clahe_applied, clahe_applied, clahe_applied])
    return processed_image

# Function to load the YOLOv8 model
@st.cache_resource()
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to display the home page
def display_home():
    st.markdown(
        """
        <style>
        .background {
            background: url('https://i.gifer.com/84lc.gif') no-repeat center center fixed;
            background-size: cover;
            padding: 50px;
            text-align: center;
            color: white;
        }
        .content {
            background-color: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="background"><div class="content"><h1>Hypertensight</h1><p>A Novel Deep Learning Augmented Ophthalmic Diagnostic System for Early Detection and Risk Stratification Reporting of Hypertensive Retinopathy</p></div></div>',
        unsafe_allow_html=True
    )
    st.header("Problem")
    st.write("Diagnosing hypertensive retinopathy (HR) is fraught with challenges primarily due to the absence of efficient diagnostic methods, which hampers early detection and treatment effectiveness. HR, a complication of hypertension affecting the retina, poses a risk of irreversible blindness if not promptly managed. Current estimates indicate that approximately 377 million individuals, representing one-third of the 1.13 billion adults with hypertension worldwide, are at risk of developing HR. The conventional approach to diagnosing HR involves manual examination of fundus images to identify subtle disease markers, a process prone to human error and time-intensive evaluations, especially in detecting early-stage symptoms. This delay in detection often results in advanced disease progression by the time symptoms become evident, compromising treatment efficacy. Moreover, the manual review process contributes to significant image review backlogs, prolonging the time to diagnosis and treatment initiation.")
    
    st.header("Solution")
    st.write("Our solution, Hypertensight, is the first autonomous AI system designed for early detection and diagnosis of hypertensive retinopathy from retinal images. Integrated into ophthalmological practices, Hypertensight uses advanced AI technologies to significantly improve the efficiency and accuracy of detection, even in early stages. By delivering rapid automated diagnostic reports, Hypertensight provides timely insights for healthcare providers, facilitating prompt intervention to mitigate vision loss risk. Employing a sophisticated AI-driven methodology, Hypertensight enhances digital retinal images through a meticulous process including image resizing, green channel extraction, and Contrast Limited Adaptive Histogram Equalization (CLAHE). Powered by YOLO v8, a robust Convolutional Neural Network trained on extensive datasets, Hypertensight excels in identifying subtle indicators of hypertensive retinopathy with high sensitivity and specificity. Healthcare providers receive immediate feedback and automated reports summarizing findings and severity assessments based on clinical scales. This streamlined process enhances diagnostic precision and empowers clinicians to optimize patient care promptly, potentially improving outcomes for hypertensive individuals at risk of vision impairment.")
    
    st.header("Our Team")
    st.write("We are a team of dedicated professionals with expertise in medical imaging and artificial intelligence, committed to improving healthcare outcomes.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://media.licdn.com/dms/image/D5603AQEtVNwteEzQ0w/profile-displayphoto-shrink_800_800/0/1713299622243?e=1723680000&v=beta&t=0Q_aW5Y41QKiLrc4u5zmPU1i_x7WLjx8yUGmztHk_pc", caption="Sarthak Ahuja", width=300)
        st.write("Meet Sarthak Ahuja, the visionary mind behind Hypertensight. With a solid foundation in Artificial Intelligence and Machine Learning, Sarthak not only conceptualized the idea but also meticulously crafted the entire platform from the ground up. His expertise, honed through an AI/ML course with the National University of Singapore and an internship in the Artificial Intelligence department at AWS, where he developed a healthcare chatbot, ensures that our diagnostic tools are not only accurate and efficient but also at the forefront of technological advancement in healthcare.")
    with col2:
        st.image("https://media.licdn.com/dms/image/D5603AQFOghhs5kVaQw/profile-displayphoto-shrink_800_800/0/1710857507676?e=1723680000&v=beta&t=NuydX9GRKHGLoQrNh5xIC5sey0GVpywau-LGR7emdAo", caption="Suhana Grewal", width=300)
        st.write("Introducing Suhana Grewal, the strategic powerhouse driving our business and partnership initiatives. With a unique background in Biology and Business, Suhana brings a deep understanding of healthcare provider needs and challenges. She has enhanced her business acumen through courses like Financial Markets by Yale University, allowing her to effectively forge partnerships and identify growth opportunities. Suhana's leadership in business development and strategic planning has been instrumental in expanding our reach and impact in the healthcare sector.")
    
# Function to display the diagnosis page
def display_diagnosis(model):
    st.header("Autonomous Hypertensive Retinopathy Detection System")
    
    # Steps and Notice
    st.subheader("Steps:")
    st.markdown("""
    1. **Capture Fundus Image:**
    Use a high-quality, in-lab fundus camera to capture an image of the patient's retina under optimal lighting conditions.
    
    2. **Save and Upload Image:**
    Save the captured image and upload it to Hypertensight.
    
    3. **Generate Analysis:**
    Click on "Analyze and Generate Report" to obtain an analysis report.
    """)
    
    st.subheader("Notice to Users")
    st.markdown("""
    - Please ensure that images are taken with a high-quality functional fundus camera.
    - Ensure that the pictures are captured under optimal lighting conditions and are saved in an appropriate aspect ratio (at least 224x224 pixels) with high quality.
    - Images sourced from the internet often have low quality and inappropriate aspect ratios, potentially rendering them unsuitable for accurate analysis.
    - We only request the patient's name to generate the report. We do not have access to any databases storing personal information, including patient names or images of their eyes, to ensure privacy.
    
    These instructions and notices aim to ensure that the input data (fundus images) are of sufficient quality and suitable for accurate analysis by Hypertensight.
    """)
    
    patient_name = st.text_input("Enter the patient's name:")
    patient_age = st.number_input("Enter the patient's Age:")
    patient_gender = st.selectbox("Enter the patient's gender?", ("Male", "Female", "Prefer not to say"))
    patient_duration = st.selectbox("How long has the patient had hypertension for?", ("<1 year", "<5 years", ">5 years"))
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        processed_image = preprocess_image(image)
        im = Image.fromarray(processed_image)
        im.save('pi.jpg')

        if st.button("Analyze and Generate Report"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                st.image(image, caption='Uploaded Image', use_column_width='always')
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                st.image(processed_image, caption='Processed Image', use_column_width='always')
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.write("Analysis Result")
            
            try:
                input_size = 224
                resized_image = cv2.resize(processed_image, (input_size, input_size))
                transformed_image = transforms.ToTensor()(resized_image).unsqueeze(0)
                results = model.predict(transformed_image)
                probe = results[0].probs.top1
                conf = results[0].probs.top1conf
                conf = conf.tolist()
                conf = round(conf, 3) * 100
                diagnosis = results[0].names[probe]
                
                if diagnosis == "optdiagnosed":
                    diagnosis_message = f"Hypertensive Retinopathy detected with {conf}% certainty"
                elif diagnosis == "opthealthy":
                    if conf > 85:
                        diagnosis_message = f"No Hypertensive Retinopathy detected with {conf}% certainty"
                    else:
                        diagnosis_message = f"No Hypertensive Retinopathy detected with {conf}% certainty, but a closer examination and subsequent consultation is advised."
                
                st.write(diagnosis_message)
                
                reportpdf(patient_name, diagnosis_message, uploaded_file, processed_image, patient_gender, patient_age, patient_duration)
                with open('report.pdf', 'rb') as report: 
                    st.download_button(
                        label = "Download Report",
                        data = report,
                        file_name = "report.pdf",
                        mime= "application/pdf"
                    )
                
            except Exception as e:
                st.error("Error during classification:")
                st.error(e)

def reportpdf(name, diagnosis_message, image, processed_image, gender, age, duration):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('helvetica', 'B', size=12)
    
    
    pdf.cell(200, 10, txt="Hypertensive Retinopathy Analysis Report", ln=True, align='C')
    pdf.ln(10)  # Add 10 units of space
    
    
    pdf.set_text_color(200, 200, 200)  # Light gray color for watermark
    pdf.set_font_size(85)
    pdf.set_font('helvetica')
    pdf.rotate(50)  # Rotate watermark text
    pdf.text(-150, 194, "Hypertensight Tech")
    pdf.rotate(0)  # Reset rotation
    
   
    pdf.set_font('helvetica', size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Date: {today}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Time: {current_time}", ln=True, align='L')
    pdf.cell(200, 10, txt=f'Patient Name: {name}', ln=True, align='L')
    pdf.cell(200, 10, txt=f'Patient Age: {age}', ln=True, align='L')
    pdf.cell(200, 10, txt=f'Patient Gender: {gender}', ln=True, align='L')
    pdf.cell(200, 10, txt=f'Duration of Hypertension: {duration}', ln=True, align='L')
    pdf.multi_cell(200, 10, txt=f'Analysis Result: {diagnosis_message}', ln=True, align='L')
    
    
    pdf.image(image, x=10, y=115, w=90, h=80)
    pdf.image('pi.jpg', x=110, y=115, w=90, h=80)
    
    pdf.set_font('helvetica', size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.text(41, 205, "Original Image")
    pdf.text(139, 205, "Processed Image")
    
   
    pdf.output("report.pdf")

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Hypertensight - Retinopathy Detection")

   
   
    with st.sidebar:
        st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
        <div style="margin-right: 10px;">
            <img src="https://firebasestorage.googleapis.com/v0/b/gr8er-ib.appspot.com/o/eyescan.png?alt=media&token=7d40ad8c-a786-4149-8c49-2cc488c782f4" width="100">
        </div>
        <div>
            <h2 style="margin: 0;">Hypertensight</h2>
            <p style="margin-bottom: 20px;">Autonomous Hypertensive Retinopathy Detection System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


        
        tabs = st.radio("", ["Home", "Diagnosis"])

    model_path = 'best.pt'  # Update with your path
    model = load_model(model_path)

    if tabs == "Home":
        display_home()
    elif tabs == "Diagnosis":
        display_diagnosis(model)

if __name__ == "__main__":
    main()
