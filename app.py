import streamlit as st
import cv2
import numpy as np
import base64
import io 
import tempfile
import tempfile
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet


st.set_page_config(page_title="Crack Analysis System", layout="wide")


def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)


set_bg("bg.jpg")



st.markdown("""
<style>
            
/* 🔥 REMOVE TOP BLACK BAR */
div[data-testid="stHeader"] {
    display: none;
}

div[data-testid="stToolbar"] {
    display: none;
}

.block-container {
    padding-top: 0rem;
}
            


/* 🔥 HIDE SIDEBAR (since you moved to top nav) */
section[data-testid="stSidebar"] {
    display: none;
}

/* NAVBAR STYLE */
div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 24px;
    background: rgba(255,255,255,0.0);
    padding: 14px 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

/* Hide radio circle completely */
div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* NAV TEXT */
div[role="radiogroup"] label {
   
    
    
    
    font-size: 50px !important;
    font-weight: 1000 !important;
    color: white !important;
    padding: 8px 18px !important;
    border-radius: 12px !important;
   
    cursor: pointer !important;
}

/* HOVER EFFECT */
div[role="radiogroup"] label:hover {
    background: rgba(0, 173, 181, 0.2);
    box-shadow: 0 0 12px rgba(0, 173, 181, 0.4);
    transform: scale(1.05);
}




/* Dark overlay */
.overlay {
    background-color: rgba(0, 0, 0, 0.65);
    padding: 50px;
    border-radius: 20px;
}

/* Glass cards */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* Text styling */
h1, h2, h3, h4, h5, h6, p, label, li, div {
    color: black !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: white;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

/* Metric color */
[data-testid="stMetricValue"] {
    color: #00FFAB;
}

</style>
""", unsafe_allow_html=True)


menu = st.radio(
    " ",
    ["HOME", "ANALYSIS", "TEAM"],
    horizontal=True
)


if menu == "HOME":

    #st.markdown('<div class="overlay">', unsafe_allow_html=True)

    st.title("CONCRETE CRACK ANALYSIS SYSTEM")

    st.markdown("""
### Intelligent Luminescent Crack Detection using Computer Vision  

Analyze structural cracks using:
- Angle-based classification  
- Real Ordinate deviation method  
- Statistical crack behavior  

---

This system helps engineers:
- Identify crack types  
- Understand structural behavior  
- Recommend repair solutions  
""")

    st.markdown('</div>', unsafe_allow_html=True)

if menu == "ANALYSIS":

    #st.markdown('<div class="overlay">', unsafe_allow_html=True)

   


    st.header("Upload Crack Image")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Beam Information")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        beam_length = st.number_input("Beam Length (mm)", min_value=0.0, value=1000.0, step=10.0)

    with col_b2:
        beam_depth = st.number_input("Beam Depth (mm)", min_value=0.0, value=300.0, step=10.0)

    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img = np.array(image)

        # Processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((3,3), np.uint8)
        processed = cv2.dilate(edges, kernel, iterations=1)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Original Image")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(processed, caption="Detected Crack")
            st.markdown('</div>', unsafe_allow_html=True)

        # Contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filtered = [c for c in contours if cv2.contourArea(c) > 300]

        if len(filtered) == 0:
            st.error("No valid crack detected")
            st.stop()

        largest_contour = max(filtered, key=lambda c: cv2.arcLength(c, False))
        points = largest_contour.reshape(-1, 2)

        # Real Ordinate Method
        points_sorted = points[points[:,1].argsort()]

        top_point = points_sorted[0]
        bottom_point = points_sorted[-1]

        xt, yt = top_point
        xb, yb = bottom_point

        x1, y1 = xt, yt
        x2, y2 = xb, yb

        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2

        num_samples = 20
        indices = np.linspace(0, len(points_sorted)-1, num_samples).astype(int)
        sampled_points = points_sorted[indices]

        distances = []
        signed_distances = []

        for (x, y) in sampled_points:
            val = A*x + B*y + C
            d = abs(val) / np.sqrt(A*A + B*B)
            distances.append(d)
            signed_distances.append(np.sign(val))

        distances = np.array(distances)
        signed_distances = np.array(signed_distances)

        # Metrics
        mean_dev = np.mean(distances)
        max_dev = np.max(distances)
        std_dev = np.std(distances)
        direction_changes = np.sum(np.diff(signed_distances) != 0)

        mid_index = len(distances) // 2
        delta1 = distances[mid_index]
        delta2 = distances[0]

        # Angle
        dy = yt - yb
        dx = xt - xb

        angle = np.degrees(np.arctan2(dy, dx))
        angle = abs(angle)

        if angle > 90:
            angle = 180 - angle

        # Classification
        if angle > 75:
            crack_type = "Flexural Crack"
        elif 30 <= angle <= 45:
            crack_type = "Shear Crack"
        else:
            crack_type = "Flexural-Shear Crack"

        # Interpretation
        if crack_type == "Flexural Crack":
            reason = f"Crack is predominantly vertical (angle ≈ {round(angle,1)}°). Local deviation (Δ1 ≈ {round(delta1,2)}, Δ2 ≈ {round(delta2,2)}) indicates minor irregularity."
        elif crack_type == "Shear Crack":
            reason = f"Crack is diagonal (angle ≈ {round(angle,1)}°). Deviation values (Δ1 ≈ {round(delta1,2)}, Δ2 ≈ {round(delta2,2)}) support shear behavior."
        else:
            reason = f"Crack starts vertical but deviates (Δ1 ≈ {round(delta1,2)}, Δ2 ≈ {round(delta2,2)}), showing combined flexural-shear behavior."

        # Visualization
        overlay = img.copy()
        cv2.line(overlay, (xt, yt), (xb, yb), (0, 0, 255), 2)

        for (x, y) in sampled_points:
            t = ((x - x1)*(x2 - x1) + (y - y1)*(y2 - y1)) / ((x2 - x1)**2 + (y2 - y1)**2)
            proj_x = int(x1 + t*(x2 - x1))
            proj_y = int(y1 + t*(y2 - y1))
            cv2.line(overlay, (x, y), (proj_x, proj_y), (255, 255, 0), 1)

        cv2.putText(overlay, f"{round(angle,1)} deg", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        st.image(overlay, caption="Final Analysis")

        # Results
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("RESULTS")
        st.write("Crack Type:", crack_type)
        st.metric("Crack Angle", f"{round(angle,1)}°")

        st.write("Δ1 (Middle Deviation):", round(delta1,2))
        st.write("Δ2 (Top Deviation):", round(delta2,2))
        st.write("Analysis:", reason)

        st.write("Mean Deviation:", round(mean_dev,2))
        st.write("Max Deviation:", round(max_dev,2))
        st.write("Std Deviation:", round(std_dev,2))
        st.write("Direction Changes:", int(direction_changes))

        st.markdown('</div>', unsafe_allow_html=True)

        # Remedies
        remedies_map = {
            "Flexural Crack": ["Epoxy injection", "Surface sealing", "Monitoring"],
            "Flexural-Shear Crack": ["Epoxy injection", "FRP wrapping", "Strengthening"],
            "Shear Crack": ["Shear reinforcement", "FRP wrapping", "Load reduction"]
        }

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🛠 Recommended Remedies")

        for r in remedies_map[crack_type]:
            st.write("✔", r)

        st.markdown('</div>', unsafe_allow_html=True)


        try:
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer)
            styles = getSampleStyleSheet()

            elements = []
            elements.append(Paragraph("Crack Analysis Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Crack Type: {crack_type}", styles["Normal"]))
            elements.append(Paragraph(f"Crack Angle: {round(angle,2)}°", styles["Normal"]))
            elements.append(Paragraph(f"Beam Length: {beam_length} mm", styles["Normal"]))
            elements.append(Paragraph(f"Beam Depth: {beam_depth} mm", styles["Normal"]))
            elements.append(Paragraph(f"Δ1 (Middle Deviation): {round(delta1,2)}", styles["Normal"]))
            elements.append(Paragraph(f"Δ2 (Top Deviation): {round(delta2,2)}", styles["Normal"]))
            elements.append(Paragraph(f"Analysis: {reason}", styles["Normal"]))
            elements.append(Paragraph(f"Mean Deviation: {round(mean_dev,2)}", styles["Normal"]))
            elements.append(Paragraph(f"Max Deviation: {round(max_dev,2)}", styles["Normal"]))
            elements.append(Paragraph(f"Std Deviation: {round(std_dev,2)}", styles["Normal"]))
            elements.append(Paragraph(f"Direction Changes: {direction_changes}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Recommended Remedies:", styles["Heading2"]))

            for r in remedies_map[crack_type]:
                elements.append(Paragraph(f"- {r}", styles["Normal"]))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                temp_image_path = tmp.name

            cv2.imwrite(temp_image_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            elements.append(Spacer(1, 12))
            elements.append(RLImage(temp_image_path, width=400, height=220))

            doc.build(elements)
            pdf_buffer.seek(0)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📄 Report")
            st.download_button(
                label="Download Report PDF",
                data=pdf_buffer,
                file_name="Crack_Report.pdf",
                mime="application/pdf"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"PDF generation error: {e}")



if menu == "TEAM":

    #st.markdown('<div class="overlay">', unsafe_allow_html=True)

    st.title("PROJECT TEAM")
    st.markdown("### GOVERNMENT ENGINEERING COLLEGE THRISSUR")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Avanthika Sundar")
        st.write("Civil Engineer, 4th Year")
        st.write("Specialization: Material Testing")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Gayathri Pradeep")
        st.write("Civil Engineer, 4th Year")
        st.write("Specialization: Structural Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Devanandha T S")
        st.write("Civil Engineer, 4th Year")
        st.write("Specialization: Concrete Technology")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Huda Shareef")
        st.write("Civil Engineer, 4th Year")
        st.write("Specialization: Structural Assessment")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
### ABOUT THE PROJECT

This system analyzes concrete cracks using:
- Angle-based classification  
- Real Ordinate Method  
- Statistical deviation analysis  

It combines civil engineering principles with computer vision.
""")

    st.markdown('</div>', unsafe_allow_html=True)