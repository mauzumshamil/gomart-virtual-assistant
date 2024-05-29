#!/usr/bin/env python
# coding: utf-8

# # End to End Chatbot using Python
# By mauzum shamil
# In[1]:


#importing the necessary python librares for this task:


# In[3]:


import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[4]:


ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


# In[5]:


#Define intents of the chatbot.


# In[6]:


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm here to assist you. How can I help?"]
    },
    {
        "tag": "general_info",
        "patterns": ["What is Gomart?", "Tell me about Gomart", "What do you do?"],
        "responses": ["Gomart is an online shopping platform where you can purchase fresh fruits, vegetables, meats, and groceries and have them delivered directly to your doorstep."]
    },
    {
        "tag": "account_creation",
        "patterns": ["How do I create an account?", "Sign up process", "Register account"],
        "responses": ["To create an account, click on the 'Sign Up' button at the top right corner of our homepage and follow the instructions to register."]
    },
    {
        "tag": "delivery_areas",
        "patterns": ["What areas do you deliver to?", "Delivery locations", "Where do you deliver?"],
        "responses": ["We currently deliver to all major areas within kochi. Please enter your zip code on our delivery page to check if we deliver to your location."]
    },
    {
        "tag": "delivery_hours",
        "patterns": ["What are your delivery hours?", "When do you deliver?", "Delivery times"],
        "responses": ["Our delivery hours are from 8 AM to 8 PM, seven days a week. You can select your preferred delivery time slot at checkout."]
    },
    {
        "tag": "customer_service",
        "patterns": ["How can I contact customer service?", "Customer support", "Need help with my order"],
        "responses": ["You can contact our customer service team via the 'Contact Us' page on our website, by calling [Customer Service Number], or by emailing [Customer Service Email]."]
    },
    {
        "tag": "placing_order",
        "patterns": ["How do I place an order?", "Order process", "Buy groceries online"],
        "responses": ["Browse our categories, add items to your cart, and proceed to checkout. Follow the on-screen instructions to complete your order."]
    },
    {
        "tag": "payment_methods",
        "patterns": ["What payment methods do you accept?", "Payment options", "How can I pay?"],
        "responses": ["We accept all major credit cards, debit cards, and digital payment methods like PayPal and Apple Pay."]
    },
    {
        "tag": "modify_order",
        "patterns": ["Can I change my order?", "Modify my order", "Edit order after placing"],
        "responses": ["Yes, you can modify or cancel your order within one hour of placing it. Please contact our customer service team for assistance."]
    },
    {
        "tag": "minimum_order",
        "patterns": ["Is there a minimum order amount?", "Minimum purchase requirement", "Small order limit"],
        "responses": ["Yes, the minimum order amount is five to ensure efficient delivery and service."]
    },
    {
        "tag": "track_order",
        "patterns": ["How can I track my order?", "Order status", "Track my delivery"],
        "responses": ["After placing your order, you will receive a tracking link via email or SMS, which you can use to monitor the status of your delivery."]
    },
    {
        "tag": "product_freshness",
        "patterns": ["Are your products fresh?", "Quality of products", "Freshness guarantee"],
        "responses": ["Yes, we source our products from trusted local suppliers and farmers to ensure they are fresh and of the highest quality."]
    },
    {
        "tag": "damaged_item",
        "patterns": ["What if I receive a damaged item?", "Wrong item delivered", "Problem with my order"],
        "responses": ["If you receive a damaged or incorrect item, please contact our customer service team immediately for a replacement or refund."]
    },
    {
        "tag": "delivery_instructions",
        "patterns": ["Can I request specific delivery instructions?", "Special delivery requests", "Custom delivery options"],
        "responses": ["Yes, you can add special delivery instructions during the checkout process to ensure your order is delivered according to your preferences."]
    },
    {
        "tag": "organic_products",
        "patterns": ["Do you offer organic products?", "Organic groceries", "Buy organic"],
        "responses": ["Yes, we offer a variety of organic fruits, vegetables, and groceries. You can filter by 'organic' in our product categories."]
    },
    {
        "tag": "food_safety",
        "patterns": ["What measures are you taking for food safety?", "Food safety protocols", "Hygiene standards"],
        "responses": ["We follow strict food safety protocols, including regular sanitization of our facilities and delivery vehicles, and contactless delivery options to ensure your safety."]
    },
    {
        "tag": "discounts",
        "patterns": ["Do you offer discounts or promotions?", "Current promotions", "Discount codes"],
        "responses": ["Yes, we regularly offer discounts and promotions. Subscribe to our newsletter or follow us on social media to stay updated."]
    },
    {
        "tag": "use_discount",
        "patterns": ["How can I use a discount code?", "Apply promo code", "Discount at checkout"],
        "responses": ["Enter your discount code at checkout in the 'Promo Code' field. The discount will be applied to your order total."]
    },
    {
        "tag": "loyalty_program",
        "patterns": ["Do you have a loyalty program?", "Customer rewards", "Loyalty points"],
        "responses": ["Yes, we have a loyalty program where you can earn points on every purchase and redeem them for discounts on future orders."]
    },
    {
        "tag": "return_policy",
        "patterns": ["What is your return policy?", "Return items", "Refund policy"],
        "responses": ["If you are not satisfied with your order, you can return it within 24 hours for a full refund or replacement. Please contact our customer service for assistance."]
    },
    {
        "tag": "recurring_delivery",
        "patterns": ["Can I schedule a recurring delivery?", "Regular delivery", "Auto-reorder"],
        "responses": ["Yes, you can schedule recurring deliveries for items you purchase regularly. Select the 'Recurring Delivery' option at checkout and choose your preferred frequency."]
    }
]


# In[7]:


#prepare the intents and train a Machine Learning model for the chatbot


# In[8]:


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# In[9]:


# writing a Python function to chat with the chatbot:


# In[10]:


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# Till now, we have created the chatbot. After running the code, you can interact with the chatbot in the terminal itself. 
# 
# To turn this chatbot into an end-to-end chatbot, we need to deploy it to interact with the chatbot using a user interface. To deploy the chatbot, I will use the streamlit library in Python, which provides amazing features to create a user interface for a Machine Learning application in just a few lines of code.

# # deploy the chatbot using Python

# In[11]:


counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
