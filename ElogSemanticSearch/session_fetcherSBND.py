#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:13:49 2024

@author: hayden
"""


import os
import requests
from ecl_api import ECL

class SessionFetcherSBND:
    def __init__(self):
        self.session = None
        self.base_url = os.environ.get('ECL_URL', 'https://dbweb1.fnal.gov:8443/ECL/sbnd/E')
        
    def create_authenticated_session(self, username, password):
        """
        Create an authenticated ECL session.
        Must be called before get_entry() or check_most_recent_entries_allCategories().
        """
        try:
            self.ecl = ECL(url=self.base_url, user=username, password=password)
            self.user     = username
            self.password = password

            # ── Verify the session actually works (equivalent of status_code == 200)
            test = self.ecl.search(limit=1)
            if not test:
                raise Exception("Authentication succeeded but session returned no data — likely bad credentials.")

            print("Authentication successful.")
            return True

        except Exception as e:
            print(f"Authentication failed: {e}")
            self.ecl = None
            return False
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        