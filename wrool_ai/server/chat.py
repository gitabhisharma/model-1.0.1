from typing import Any
import pandas as pd
import numpy as np
import django as dj
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from .models import ChatMessage, UserProfile
from .utils import get_chat_response, log_chat_message

import os
from dotenv import load_dotenv
import openai

from flask import Flask, template

app = Flask(__name__)
@app.route("/")
async def template():
    return template("index.html")

# from deepseek_api import DeepSeek  # Hypothetical DeepSeek package

# Mock DeepSeek client if package is not available
class DeepSeek:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def chat(self, model, messages, temperature, max_tokens):
        # Return a mock response for development/testing
        return {
            "choices": [
                {"message": {"content": "This is a mock DeepSeek response."}}
            ]
        }
from flask import Flask, request, jsonify, session
from functools import wraps
import time
import hashlib
import json
from datetime import datetime, timedelta
import uuid
from werkzeug.utils import secure_filename
import tempfile

# Load environment variables
load_dotenv()

# Initialize APIs
openai.api_key = os.getenv("OPEN_KEY")
deepseek_client = DeepSeek(api_key=os.getenv("sk-615585cb445d419e86e37364ef848052"))

app = Flask(__name__)
app.secret_key = os.getenv("460d7246246b1ec2d118618a4a6a62b8f3234f96f0981bcaa3ebbfc67b236953", "supersecretkey")
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Rate limiting setup
RATE_LIMIT = 10  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Cache setup
CACHE_EXPIRY = 3600  # 1 hour in seconds

class RateLimiter:
    def __init__(self):
        self.requests = {}

    def check_rate_limit(self, user_id):
        now = time.time()
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Remove old requests
        self.requests[user_id] = [t for t in self.requests[user_id] if t > now - RATE_LIMIT_WINDOW]
        
        if len(self.requests[user_id]) >= RATE_LIMIT:
            return False
        
        self.requests[user_id].append(now)
        return True

class ResponseCache:
    def __init__(self):
        self.cache = {}
    
    def get_cache_key(self, messages, api_choice):
        """Generate a consistent cache key from the conversation context"""
        key_str = json.dumps({"messages": messages, "api": api_choice}, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        """Get cached response if it exists and isn't expired"""
        if key in self.cache:
            cached_time, response = self.cache[key]
            if time.time() - cached_time < CACHE_EXPIRY:
                return response
            del self.cache[key]
        return None
    
    def set(self, key, response):
        """Cache a response"""
        self.cache[key] = (time.time(), response)

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'api_responses': {
                'openai': {'count': 0, 'total_time': 0, 'errors': 0},
                'deepseek': {'count': 0, 'total_time': 0, 'errors': 0},
                'fallback': {'count': 0}
            },
            'cache_hits': 0
        }
    
    def record_api_call(self, api_name, duration, success=True):
        self.metrics['total_requests'] += 1
        if api_name not in self.metrics['api_responses']:
            self.metrics['api_responses'][api_name] = {'count': 0, 'total_time': 0, 'errors': 0}
        
        self.metrics['api_responses'][api_name]['count'] += 1
        self.metrics['api_responses'][api_name]['total_time'] += duration
        
        if not success:
            self.metrics['api_responses'][api_name]['errors'] += 1
    
    def record_cache_hit(self):
        self.metrics['cache_hits'] += 1
    
    def get_metrics(self):
        metrics = self.metrics.copy()
        
        # Calculate average response times
        # for api in metrics['api_responses']:
        #     if metrics['api_responses'][api]['count'] > 0:
        #         metrics['api_responses'][api]['avg_time'] = (
        #             metrics['api_responses'][api]['total_time'] /
        #             metrics['api_responses'][api]['count']
        #         )
        
        return metrics

class Chatbot:
    def __init__(self):
        self.sessions = {}
        self.cache = ResponseCache()
        self.rate_limiter = RateLimiter()
        self.metrics = PerformanceMetrics()
        self.max_history = 15  # Keep last 15 messages in memory
        self.file_processing_limit = 5 * 1024 * 1024  # 5MB
    
    def get_session(self, session_id):
        """Get or create a user session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'conversation_history': [],
                'created_at': time.time(),
                'last_activity': time.time()
            }
        else:
            self.sessions[session_id]['last_activity'] = time.time()
        return self.sessions[session_id]
    
    def cleanup_sessions(self):
        """Remove inactive sessions (older than 24 hours)"""
        now = time.time()
        inactive_sessions = [sid for sid, data in self.sessions.items() 
                           if now - data['last_activity'] > 86400]  # 24 hours
        for sid in inactive_sessions:
            del self.sessions[sid]
    
    def add_to_history(self, session_id, role, content, files=None):
        """Add a message to the conversation history"""
        session = self.get_session(session_id)
        message = {"role": role, "content": content}
        if files:
            message["files"] = files
        session['conversation_history'].append(message)
        
        # Trim history if it gets too long
        if len(session['conversation_history']) > self.max_history * 2:
            session['conversation_history'] = session['conversation_history'][-self.max_history:]
    
    def process_file(self, file):
        """Process uploaded file (placeholder for actual implementation)"""
        # In a real implementation, you would:
        # 1. Check file type (PDF, TXT, etc.)
        # 2. Extract text (using PyPDF2 for PDFs, etc.)
        # 3. Return relevant content
        # For now, we'll just return a placeholder
        return f"File processed: {file.filename} (size: {file.content_length} bytes)"
    
    def generate_response(self, session_id, user_input, api_choice="openai", files=None):
        """Generate a response using the selected API with fallback"""
        session = self.get_session(session_id)
        
        # Process files if any
        processed_files = []
        if files:
            for file in files:
                if file.content_length > self.file_processing_limit:
                    return "File size exceeds limit (5MB)"
                processed_files.append(self.process_file(file))
        
        # Add user message to history
        self.add_to_history(session_id, "user", user_input, processed_files)
        
        # Check cache first
        cache_key = self.cache.get_cache_key(session['conversation_history'], api_choice)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            self.metrics.record_cache_hit()
            self.add_to_history(session_id, "assistant", cached_response)
            return cached_response
        
        # Try primary API, with fallback to the other if it fails
        response = None
        start_time = time.time()
        
        try:
            if api_choice.lower() == "openai":
                try:
                    response = self._get_openai_response(session['conversation_history'])
                    self.metrics.record_api_call("openai", time.time() - start_time)
                except Exception as e:
                    print(f"OpenAI error: {str(e)}")
                    self.metrics.record_api_call("openai", time.time() - start_time, False)
                    # Fallback to DeepSeek
                    response = self._get_deepseek_response(session['conversation_history'])
                    self.metrics.record_api_call("deepseek", time.time() - start_time)
                    self.metrics.record_api_call("fallback", 0)
            
            elif api_choice.lower() == "deepseek":
                try:
                    response = self._get_deepseek_response(session['conversation_history'])
                    self.metrics.record_api_call("deepseek", time.time() - start_time)
                except Exception as e:
                    print(f"DeepSeek error: {str(e)}")
                    self.metrics.record_api_call("deepseek", time.time() - start_time, False)
                    # Fallback to OpenAI
                    response = self._get_openai_response(session['conversation_history'])
                    self.metrics.record_api_call("openai", time.time() - start_time)
                    self.metrics.record_api_call("fallback", 0)
            
            else:
                response = "Invalid API choice. Please select 'openai' or 'deepseek'."
            
            # Cache the response
            if response and not any(err in response.lower() for err in ["error", "fail"]):
                self.cache.set(cache_key, response)
            
            self.add_to_history(session_id, "assistant", response)
            return response
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def _get_openai_response(self, messages):
        """Get response from OpenAI API"""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def _get_deepseek_response(self, messages: Any) -> str:
        """Get response from DeepSeek API"""
        # Note: This may vary based on DeepSeek's actual API implementation
        response = deepseek_client.chat(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']

# Initialize the chatbot
chatbot = Chatbot()

# Helper decorators
def require_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not chatbot.rate_limiter.check_rate_limit(session.get('session_id')):
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        return f(*args, **kwargs)
    return decorated_function

# Flask API endpoints
@app.route('/chat', methods=['POST'])
@require_session
@rate_limit
def chat():
    data = request.form if request.files else request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_message = data.get('message', '')
    api_choice = data.get('api', 'openai')  # Default to OpenAI
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Handle file uploads if any
    files = []
    if request.files:
        files = list(request.files.values())
    
    response = chatbot.generate_response(
        session['session_id'],
        user_message,
        api_choice,
        files
    )
    
    return jsonify({"response": response})

@app.route('/history', methods=['GET'])
@require_session
def get_history():
    session_data = chatbot.get_session(session['session_id'])
    return jsonify({"history": session_data['conversation_history']})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # In production, you'd want to secure this endpoint
    chatbot.cleanup_sessions()
    metrics = chatbot.metrics.get_metrics()
    metrics['active_sessions'] = len(chatbot.sessions)
    return jsonify(metrics)

@app.route('/reset', methods=['POST'])
@require_session
def reset_session():
    if 'session_id' in session and session['session_id'] in chatbot.sessions:
        del chatbot.sessions[session['session_id']]
    session.pop('session_id', None)
    return jsonify({"status": "session reset"})

if __name__ == '__main__':
    app.run(debug=True)


 