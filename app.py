import streamlit as st 
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipline import CustomData , PredictPipeline

# 1. Structured dataset mapping Brands to their respective Models
brand_model_mapping = {
    "OPPO": [
        'A53', 'A12', 'A53s 5G', 'A33', 'A31', 'A74 5G', 'A11K', 'F17 Pro', 'A54',
        'Reno6 5G', 'F17', 'A16', 'Reno5 Pro 5G', 'A15', 'A74 5G BLACK', 'Reno6 Pro 5G',
        'Reno2 F', 'Reno3 Pro', 'A15s', 'A15S', 'F19 Pro', 'F19 Pro+ 5G', 'F19', 'A7',
        'F5', 'A5s', 'Reno', 'Reno2 Z', 'F15', 'K1', 'A9 2020', 'Find X',
        'Reno4 Pro Special Edition', 'F11', 'A5', 'Reno4 Pro', 'A5 2020',
        'F3 Deepika Padukone Limited Edition', 'F1 Plus', 'A83 2018 Edition', 'F3',
        'A3s', 'A83', 'K3', 'F9', "A52", 'A71 New Edition', 'Neo 7 4G', 'A71', 'A57',
        'A1K', 'A37f', 'F9 Pro', 'F19s', 'A71k', 'A9', 'F3 Plus', 
        'Rohit Sharma Limited Edition', 'F17 PRO', 'Hardik Pandya Limited Editon',
        'Ravichandran Ashwin Limited Edition', 'F11 Pro', 'R1 R829', 'F5 Youth', 'F7',
        'Neo 5', 'F1', 'Reno2', 'R17', 'F11 Pro Marvels Avengers Limited Edition',
        'Reno 10x Zoom', 'F1S', 'N5111'
    ],
    "HTC": ['U11+', 'Wildfire X', '3'],
    "IQOO": ['3', '7'],
    "Google Pixel": ['4a', '3a XL', '3 XL', '3a', 'XL', '2 XL', '2'],
    "LG": [
        'W11', 'Velvet Dual Screen', 'Q60', 'L90 Dual', 'W31', 'W31 Plus', 'K42', 
        'W30 Pro', 'Candy K9', 'K9 4G LTE', 'W10', 'K-10', 'W41', 'Q Stylus+', 
        'W10 Alpha', 'W30', 'W30 Plus', 'G7+ ThinQ', 'Stylus 2', 'Max X160', 'V20', 
        'G8X', 'V40 ThinQ', 'K7i', 'Nexus4 E960', 'G7 ThinQ', 'Nexus 5X', 'G5', 
        'Q6+', 'Spirit 4G LTE', 'Velvet', 'W41 Plus', 'Wing', 'W41 Pro', 
        'Optimus L70', 'K8', 'G Pro 2', 'V30+', 'G2 D802', 'Q Stylus', 'Q6', 'G6', 
        'K10 2017', 'V20a', 'K10 K420DS', 'Q7', 'X Power', 'Stylus 2 Plus', 'K-7', 
        'G3 Beat', 'G8s ThinQ', 'G4', 'Stylus 3', 'L Bello', 'G4 Stylus 4G LTE', 
        'X Cam', 'L 80 Dual', 'G3 Stylus', 'Q7+', 'X Screen K500I', 'Spirit'
    ],
    "ASUS": [
        'ROG Phone 5', 'ROG Phone 3', 'Zenfone Max Pro M1', 'Zenfone 2 Laser ZE500KL', 
        'ZenFone Max M2', 'ZenFone 5Z', '6Z', 'Zenfone Go 4.5', 'Zenfone Go 5.5', 
        'Zenfone Go 3rd Gen', 'Zenfone Go 2nd Gen', 'Zenfone Go', 'ZenFone Lite L1', 
        'Zenfone 4 Selfie', 'ROG Phone II', 'Zenfone GO', 'Zenfone Go 5.0', 
        'ZenFone Max Pro M2', 'Zenfone Go 5.0 LTE 2nd Gen', 'Zenfone C', 'Zenfone Go 5.0 LTE'
    ],
    "realme": [
        'Narzo 30 5G', 'C20', 'C21Y', '8i', 'C25s', '8s 5G', 'C25Y', 'C21', 
        'Narzo 30', 'Narzo 30A', '8 5G', '8 Pro', 'C15', '8', 'C12', 'C11', '7', 
        'X7 5G', 'Narzo 30 Pro 5G', 'C3', 'C15 Qualcomm Edition', '3i', '5 Pro', 
        '7i', 'X3 SuperZoom', 'X50 Pro 5G', 'X3', 'GT 5G', 'X50 Pro', 'C2', '6', 
        'X7 Pro 5G', 'GT Master Edition', '6i', '1', 'X2', '5i', 'X', 'Narzo 50i', 
        'Narzo 50A', 'C25', '7 Pro', 'Narzo 20 Pro', 'Narzo 20A', 'Narzo 20', 
        'Narzo 10A', 'Narzo 10', '6 Pro', 'U1', 'X2 Pro', '5s', 'Max', 'Max Pro'
    ],
    "GIONEE": [
        'F8 Neo', 'F103 Pro', 'F205 Pro', 'A1', 'S11 Lite', 'F11', 'F10 Plus', 
        'L800', 'Pioneer P5W', 'S96', 'L700', 'P5L', 'P5 Mini', 'P5_W', 'F9', 'X-1', 
        'M5 Lite 4G', 'P7', 'A1 Plus', 'Pioneer P4', 'Pioneer P3S', 'X1s', 'F10', 
        'F205', 'A1 Lite', 'Elife S7', 'Marathon M5 lite CDMA', 'Marathon', 
        'Marathon M3', 'M7 Power', 'S10 Lite', 'S6S', 'M3', 'S Plus', 'Elife E8', 
        'F9 Plus', 'Pioneer P4S', 'Marathon M5 Lite', 'P7 Max', 'M7', 
        'F103 3GB Version', 'S6', 'S6 Pro', 'F103', 'S11', 'V4S', 'Marathon M5 Plus', 
        'G3', 'F103 3Gb Version', 'Elife E7 Mini', 'Ctrl V4S', 'M2', 'Elife S5.1', 
        'Elife E3', 'Pioneer P6'
    ],
    "Nokia": [
        '105 DS 2020', 'TA-1010/105', '110 TA-1302 DS', '150 DS 2020', 
        '150 TA-1235 DS', '6310', '105 SS 2021', 'C01 Plus', '125 DS', '5310ds', 
        '3310 DS 2020', '3.4', '125 TA-1253 DS', 'C20 Plus', 'G20', '5.4', 
        'TA-1174 / TA-1299', '2.4', 'G10', '125 DS 2020', '5310 TA-1212 DS', 
        '216 DS 2020', '215 4G DS 2020', '225 4g ds', '225 4G DS 2020', '2.3', 
        '215 4G DS', '9', '150', '105 DS', '215', '130', '108 Dual SIM', 
        '107 Dual SIM', '150/150 DS', '3.2', '5.1 Plus', 'Ta -1010/105', '105', 
        '6.1', '2.1', '5', '112', '3.1', '5.1', '2.2', 'Lumia 920', '3.1 Plus', 
        '110', '225', '106', '8 Sirocco', 'RM-1172 / Nokia 230 DS', '5.3', 
        '6.1 Plus', '8.1', '7.2', '7.1', 'X2 Dual SIM', '7 Plus', 'Asha 206', 
        '222', '6.2', '4.2', 'Asha 502', '515'
    ],
    "Apple": [
        'iPhone SE', 'iPhone XR', 'iPhone 12 Mini', 'iPhone 13 Pro', 'iPhone 11', 
        'iPhone 12', 'iPhone 13', 'iPhone 13 Mini', 'iPhone 12 Pro', 
        'iPhone 11 Pro Max', 'iPhone 8 Plus', 'iPhone 11 Pro', 'iPhone XS', 
        'iPhone 8', 'iPhone 13 Pro Max', 'iPhone 6 Plus', 'iPhone XS Max', 
        'iPhone 7 Plus', 'iPhone 6s', 'iPhone 7', 'iPhone 6', 'iPhone 6s Plus', 
        'iPhone X', 'iPhone 6s'
    ],
    "SAMSUNG": [
        'Galaxy F22', 'Galaxy F12', 'M31s', 'M02s', 'Galaxy M02', 'Galaxy A12', 
        'Galaxy A22', 'Galaxy A22 5G', 'Galaxy M12', 'M31', 'M21 2021 Edition', 
        'Galaxy A21s', 'GALAXY M31S', 'M32 5G', 'Galaxy M31', 'Galaxy Z Flip3 5G', 
        'Galaxy M01', 'Galaxy A03s', 'Galaxy F02s', 'Galaxy A72', 'Galaxy M31s', 
        'Galaxy M32', 'Galaxy A51', 'Galaxy F62', 'Galaxy A52s 5G', 
        'Galaxy M21 2021 Edition', 'Galaxy M11', 'Galaxy S10 Lite', 'Galaxy F41', 
        'Galaxy A52', 'Galaxy A32', 'S20 FE 5G', 'Galaxy S10', 'M01 core', 
        'GALAXY M51', 'Galaxy Note 20', 'Galaxy Note10 Lite', 'Galaxy A71', 
        'Galaxy A70s', 'Galaxy Z Fold3 5G', 'Galaxy M32 5G', 'Galaxy A50s', 
        'Galaxy S9 Plus', 'Galaxy A6', 'Galaxy A6+', 'Galaxy On7', 'Galaxy A70', 
        'Galaxy A31', 'M21 2021 Edition', 'Galaxy Note 20 Ultra 5G', 'Galaxy M01s', 
        'Metro 350', 'Galaxy A20s', 'Galaxy J7 Prime', 'Galaxy S20 FE', 
        'Galaxy A30s', 'Galaxy A7', 'Galaxy J2 Core', 'Galaxy J6', 'Galaxy A20', 
        'Galaxy A10s', 'Galaxy M21', 'Galaxy J2 2018', 'Galaxy J7 Nxt', 'Galaxy M30', 
        'Galaxy M42', 'Galaxy J7 Duo', 'Galaxy A9', 'Galaxy J6 Plus', 'Galaxy Fold 2', 
        'Galaxy J7 - 6 New 2016 Edition)', 'Galaxy S20 Ultra', 'Galaxy Note 9', 
        'Galaxy A10', 'Galaxy M40', 'Galaxy J2', 'Galaxy A50', 'Galaxy J2-2017', 
        'Galaxy Grand 2', 'Galaxy S10 Plus', 'Galaxy J4', 'Galaxy A30', 'Galaxy M51', 
        'Galaxy J5 Prime', 'Galaxy S6 Edge', 'Galaxy S21 Ultra', 'Galaxy J8', 
        'Galaxy A7-2017', 'Galaxy J4 Plus', 'Galaxy A8 Plus', 'Galaxy A80', 
        'Galaxy M30s', 'Galaxy S5', 'Galaxy J7 Prime 2', 'Galaxy S21 Plus', 
        'Galaxy S21', 'Galaxy M42 5G', 'Galaxy On8', 'Galaxy S20+', 'Galaxy A2 Core', 
        'Galaxy M10', 'M31 Prime', 'Galaxy S9', 'Galaxy A7 2016 Edition', 'Metro XL', 
        'Galaxy M10S', 'Galaxy M20', 'Galaxy J5', 'Galaxy On6', 'Galaxy Note 5', 
        'Galaxy S4 Mini', 'Galaxy A9 Pro', 'Galaxy J2 - 2016', 'Galaxy J2 Pro', 
        'Galaxy Note 10 Plus', 'Galaxy S8', 'Galaxy J1', 'Galaxy S20', 'Galaxy Z Flip', 
        'Galaxy Alpha', 'Galaxy Core Prime G361 Dual Sim - White', 'Galaxy E7', 
        'Galaxy Note 8', 'Galaxy A5-2017', 'Tizen Z3', 'Galaxy On Nxt', 'Galaxy M30S', 
        'Galaxy J2 Ace', 'Galaxy S4', 'Galaxy Note 3', 'Galaxy A3', 'Galaxy Note 3 Neo', 
        'Galaxy A8', 'Galaxy A5 2016 Edition', 'Galaxy J7', 'Galaxy Core', 
        'Galaxy Note 10', 'Galaxy Grand Neo', 'Galaxy Note 4', 'On7 Pro', 
        'Galaxy S10e', 'Galaxy S7 Edge', 'Galaxy A8 Star', 'Galaxy S3 Neo', 
        'Galaxy C7 Pro', 'Galaxy Folder 2', 'Galaxy J5 - 6 New 2016 Edition)', 
        'Metro 360', 'Grand Prime 4G', 'Fold 2 5G', 'Galaxy Core Prime', 'On5 Pro', 
        'Galaxy A5', 'Galaxy Grand Neo Plus', 'Galaxy Note Edge', 'Galaxy S6', 
        'Grand Prime', 'Galaxy S8 Plus', 'Rex 60', 'Galaxy S7', 'Galaxy E5', 
        'Sm-G361Hhadins', 'Guru FM Plus', 'Galaxy Grand Quattro', 'Galaxy Grand I9082', 
        'Galaxy Mega 5.8', 'Galaxy Note 5 Dual', 'GURU FM PLUS', 'Galaxy S4 Zoom', 
        'Galaxy Grand Prime 4g', 'S7 Edge', 'Galaxy S6 Edge+'
    ],
    "Lenovo": [
        'A6600d40', 'K8 Plus', 'K3 Note', 'S930', 'A2010', 'Vibe Shot', 'A7700', 
        'K8 Note', 'K9', 'VIBE P1m', 'Z2 Plus', 'Vibe K5 Note', 'B', 'S850', 
        'A6600', 'A5000', 'A6600 Plus', 'A536', 'A7000', 'K10 Note', 'A7000 Turbo', 
        'Vibe K5 Plus', 'K6 Note', 'K6 Power', 'X2-AP', 'K10 Plus', 'P770', 
        'VIBE P1', 'K4 Note', 'S660', 'Vibe Z2 Pro', 'K9 Note', 'P780', 'A850', 
        'P2', 'A328', 'A6 Note', 'Vibe P1 Turbo', 'A1000', 'A6000 Plus', 'S560', 
        'Sisley S60', 'P70', 'S90 Or Sisley S90', 'A6000 Shot', 'Vibe S1', 
        'K6 POWER', 'K3 Note Music', 'ZUK Z1'
    ],
    "Motorola": [
        'E7 Power', 'Edge 20 Fusion', 'G10 Power', 'G60', 'G40 Fusion', 
        '2nd Generation', 'Edge 20', 'G8 Power Lite', 'X4', 'Razr 5G', 'E5', 
        'Razr', 'X Play', 'C Plus', 'One Vision', 'G30', 'G9 Power', 'G 2nd Generation'
    ],
    "POCO": ['Z2', 'X4', 'M'],
    "vivo": ['V20'],
    "Xiaomi": ['G6', 'M3'],
    "Infinix": ['G5', '3']
}
# Set up page configuration
st.set_page_config(page_title="Smartphones sales price prediction", layout="centered")

# Title and Description
st.title("Smartphones sales price prediction")
st.write("Enter the details below to predict smartphones selling price.")

st.subheader("Smartphone Context Details")

Brand = st.selectbox(
    label="Select Brand",
    options=list(brand_model_mapping.keys())
)

available_models = brand_model_mapping[Brand]

Model = st.selectbox(
    label=f"Select {Brand} Model",
    options=available_models
)

with st.form("prediction_form"):

    Color = st.selectbox("color",
        options=['Black','White/Cream', 'Blue', 'Silver', 'Green', 'Purple', 'Gold', 'Aurora',
            'White', 'Orange', 'Red', 'Starry Night', 'Grey', 'Pink', 'Brown', 'Ice',
            'Black & Blue', 'Black & Gold', 'Illusion Sky', 'Yellow', 'Nebula',
            'Diamond Sapphire', 'Purple/Pink', 'Fjord', 'Glacier', 'Polar Night',
            'Gold/Beige', 'Dusk', 'Night', 'Copper/Bronze', 'Black & Copper',
            'Blue & Silver', 'Baltic', 'Silver/Grey', 'Cyan/Teal', 'White & Copper',
            'Silver/White', 'Black/Grey', 'Cloud Navy', 'Aura Glow', 'Sprite',
            'Frosted Pearl', 'Sapphire Gradient', 'Dark Pearl', 'Pastel Sky',
            'Smoky Sangria', 'Rich Cranberry', 'Metallic Sage', 'Midday Dream' ,
            'Diamond Glow', 'Diamond Flare', 'Prism Magic', 'Crystal Symphony',
            'Sunset Jazz', 'Sunset Melody', 'Sunset Dazzle', 'Fantastic Rainbow' ,
            'Pacific Sunrise','Sunrise Flare', 'Dark Nebula' ,'Dark Night','Neon Spark',
            'Aurora Dawn','Celestial Magic','Pacific Pearl' ,'Diamond Dazzle',
            'Ocean Wave' ,'Celestial Snow', 'Nordic Secret', 'Heart Of Ocean',
            'Moonlight Jade', 'Rainbow Fantasy', 'Glowing Galaxy', 'Azure Glow',
            'Starry Glow'],
    )
    
    Rating = st.slider("rating", 
                        min_value=0.0, max_value=5.0, step=0.1
    )
    
    Storage = st.selectbox("Storage",
        options=[64, 128, 32, 256, 16, 8, 4, 512, 10, 100, 129, 130],
    )
    
    Memory = st.selectbox(
        "memory", options=[4, 6, 3, 8, 2, 12, 1, 1.5, 16, 18, 64, 32, 46, 0.5, 30]
    )
    
    Original_Price = st.number_input(
        "Original Price", min_value=1149, max_value=179900
    )
    
    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Performance")

# Handle form submission and prediction
if submit_button:
    # Form submission can safely pull 'brand' and 'model' from outside widgets
    data = CustomData(
        Brand = Brand,
        Model = Model,
        Color = Color,
        Rating = Rating,
        Storage = float(Storage),
        Memory = float(Memory),
        Original_Price = float(Original_Price),
    )

    # Convert to DataFrame
    pred_df = data.get_data_as_data_frame()

    print(pred_df)
    print("Before Prediction")

    # Run the prediction pipeline
    try:
        predict_pipeline = PredictPipeline()
        results = np.expm1(predict_pipeline.predict(pred_df))
        st.success(f"### Predicted Price: ₹{results[0]:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Ensure all background model artifacts exist.")
        st.exception(e) # Visible block during debugging, remove in production
