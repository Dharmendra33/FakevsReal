import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import mysql.connector
    
    # Function to create a MySQL connection
    def create_connection():
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="R4rachit.s@123",
            database="user"
        )
    
    
    # Function to create a registration table if not exists
    def create_table(cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registrations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
    
    
    # Function to insert registration data into the MySQL database
    def insert_data(cursor, username, email, password):
        cursor.execute("""
            INSERT INTO registrations (username, email, password)
            VALUES (%s, %s, %s)
        """, (username, email, password))
    
    
    # Function to check login credentials
    def check_login(cursor, username, password):
        cursor.execute("""
            SELECT * FROM registrations
            WHERE username = %s AND password = %s
        """, (username, password))
        return cursor.fetchone() is not None
    
    
    # Function to preprocess symbolic input
    def _preprocess_symbolic_input(x, data_format, mode):
        if mode == 'tf':
            x /= 127.5
            x -= 1.
            return x
    
        # Add a check for zero before division
        divisor = 127.5
        x_max = 1.0 / divisor
        x_min = -1.0 / divisor
    
        # Check if x is within valid bounds
        mask = tf.math.logical_and(tf.math.greater_equal(x, x_min), tf.math.less_equal(x, x_max))
    
        # Clip values outside the bounds
        x = tf.where(mask, x, 0.0)
    
        # Avoid division by zero
        x = tf.where(mask, x / divisor, x)
    
        return x
    
    
    # Streamlit app
    def main():
        st.title("User Authentication App")
    
        # Sidebar navigation
        page = st.sidebar.radio("Navigation", ["Register", "Login", "Homepage"])
    
        # Session state to keep track of authentication status
        session_state = st.session_state
        if 'authenticated' not in session_state:
            session_state.authenticated = False
    
        if page == "Register":
            st.header("Registration Page")
    
            # Input fields for registration
            reg_username = st.text_input("Username")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
    
            # Register button
            if st.button("Register"):
                # Create a MySQL connection and cursor
                connection = create_connection()
                cursor = connection.cursor()
    
                # Create the registration table if not exists
                create_table(cursor)
    
                # Insert registration data into the MySQL database
                insert_data(cursor, reg_username, reg_email, reg_password)
    
                # Commit changes and close the connection
                connection.commit()
                connection.close()
    
                st.success("Registration successful!")
    
        elif page == "Login":
            st.header("Login Page")
    
            # Input fields for login
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
    
            # Login button
            if st.button("Login"):
                # Create a MySQL connection and cursor
                connection = create_connection()
                cursor = connection.cursor()
    
                # Check login credentials
                if check_login(cursor, login_username, login_password):
                    st.success("Login successful!")
                    session_state.authenticated = True
                else:
                    st.error("Invalid username or password.")
    
                # Close the connection
                connection.close()
    
        elif page == "Homepage":
            st.header("Homepage")
    
            # Only show the homepage if the user is authenticated
    
            if not session_state.authenticated:
                st.error("Access denied. Please log in.")
                st.stop()
    
            # Add content for the homepage/dashboard
            trained_model = load_model('C:/Users/shahr/PycharmProjects/pythonProject2/model_cp.h5')
            from tensorflow.keras.models import load_model
            load_model(trained_model, "model_cp.h5")
            # File uploader for selecting an image file
            uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
    
            # Check if a file is uploaded
            if uploaded_file is not None:
                # Process the uploaded file (you can replace this with your own logic)
                pil_image = Image.open(uploaded_file)
    
                # Convert the image to a NumPy array
                img_array = tf.keras.preprocessing.image.img_to_array(pil_image)
    
                st.image(pil_image, caption="Uploaded Image.", use_column_width=True)
    
                # Resize the image to (224, 224) before preprocessing
                img_array = tf.image.resize(img_array, [224, 224])
    
                # Preprocess the image for model prediction
                img_array = _preprocess_symbolic_input(img_array, data_format=None, mode='tf')
                img_array = tf.expand_dims(img_array, axis=0)
    
                # Make predictions
                predictions = model.predict(img_array)
    
                # Display the results
                st.write("Model Predictions:")
                st.write(predictions)
    
                # Interpret the predictions (you might need to adjust this based on your model and task)
                # For simplicity, assuming binary classification (cat or not cat)
                if predictions[0][0] > 0.5:
                    st.write("Predicted: Cat")
                else:
                    st.write("Predicted: Not a Cat")
    
            st.write("Welcome to the homepage!")
    
    
    main()  
