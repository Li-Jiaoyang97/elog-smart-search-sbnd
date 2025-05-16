#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 05:59:49 2024

@author: hayden
"""
import os
import requests

class ElogFetcher:
    def __init__(self):
        self.username = os.environ.get('USERNAME')
        self.password = os.environ.get('PASSWORD')
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
    
    def get_entry(self, entry_id):
        if self.session is None:
            print("Session not authenticated.")
            return None
    
        entry_url = f"{self.base_url}api/get/entry?entryId={entry_id}&format=JSON"
        response = self.session.get(entry_url)
        # Check the response status code
        entry_details = {}
        if response.status_code == 200:
            try:
                # Parse JSON data
                json_data = response.json()
    
                # Since the data is a dictionary, directly access its values
                if isinstance(json_data, dict):
                    # Extract various fields from the dictionary. This is not all the data you can pull, but this is probably where you should start.
                    entry_details['id'] = json_data.get('id', 'No ID Provided')
                    entry_details['logs'] = [log['name'] for log in json_data.get('logs', []) if 'name' in log]
                    entry_details['categories'] = [category['name'] for category in json_data.get('categories', []) if 'name' in category]
                    entry_details['creation_date'] = json_data.get('creationDate', 'No Date Found')
                    entry_details['text'] = json_data.get('text', 'No Text Provided')
    
                    # Handle multiple comments, each with 'text' and 'modifiedDate'
                    entry_details['comments'] = [{
                        'text': comment.get('text', 'No Comment Text Provided'),
                        'modified_date': comment.get('modifiedDate', 'No Date Provided')
                    } for comment in json_data.get('comments', [])]  # Ensure comments is a list
    
                    return entry_details
                else:
                    print("JSON data is not a dictionary as expected.")
                    
            except ValueError:
                print("Error: Failed to decode JSON.")
        else:
            print(f"Failed to retrieve data, status code: {response.status_code}")
            print("Response Text:", response.text)  # Optionally display the raw response text. Sometimes we get an error thrown by the ELOG
   
       
    def check_most_recent_entry(self):
        """
        Check the most recent entry ID from the Elog. Good if you need to pull a range of Elog ID's
        """
        if self.session is None:
            print("Session not authenticated.")
            return None
        url = f"{self.base_url}api/search/entries?entryLimit=1&format=JSON"
        response = self.session.get(url)
            # Check the response status code
        if response.status_code == 200:
            try:
                # Parse JSON data
                json_data = response.json()
                
                # Ensure json_data is a list and contains elements
                if isinstance(json_data, list) and json_data:
                    entry = json_data[0]  # Extracting elements from the first item in the list
                    
                    # Extract various fields from the first item
                    id_value = entry['id']
                
                else:
                    print("JSON data is empty or not a list.")
                    
            except ValueError:
                print("Error: Failed to decode JSON.")
        else:
            print(f"Failed to retrieve data, status code: {response.status_code}")

        return id_value
    

    
    
    
    
    
    
   
