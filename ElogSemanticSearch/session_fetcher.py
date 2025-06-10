#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:13:49 2024

@author: hayden
"""


import os
import requests

class SessionFetcher:
    def __init__(self):
        self.session = None
        self.base_url = "https://www-bd.fnal.gov/Elog/"
        
    def create_authenticated_session(self, username, password):
        """
        Authenticate with the Elog server and return a session object with the authenticated cookie. THE OTHER METHODS WON"T WORK IF YOU DON'T USE THIS FIRST. YOU'll JUST PULL THE LOGIN SCREEN.
        """
        context_path = self.base_url              
        auth_url = f"{context_path}authentication/authenticate"        

        self.session = requests.Session()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Charset": "UTF-8",
            "Accept": "application/xml",
            "Referer": "Elog API"
        }

        data = f"userName={username}&passWord={password}"
        response = self.session.post(auth_url, headers=headers, data=data)

        if response.status_code == 200:
            print("Authentication successful.")
            return True
        else:
            print(f"Authentication failed with status code: {response.status_code}")
            return False
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        