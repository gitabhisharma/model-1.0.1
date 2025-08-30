document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loginModal = document.getElementById('login-modal');
    const registerModal = document.getElementById('register-modal');
    const showRegisterLink = document.getElementById('show-register');
    const showLoginLink = document.getElementById('show-login');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const appContainer = document.getElementById('app-container');
    const toast = document.getElementById('toast');
    const newChatBtn = document.getElementById('new-chat-btn');
    const chatHistory = document.getElementById('chat-history');
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const fileInput = document.getElementById('file-input');
    const fileBtn = document.getElementById('file-btn');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    const modelSelector = document.getElementById('model-selector');
    const chatTitle = document.getElementById('chat-title');
    const userNameElement = document.getElementById('user-name');
    const userEmailElement = document.getElementById('user-email');
    const userAvatar = document.getElementById('user-avatar');

    // sidebar click functions
//    const menuToggle = document.getElementById("menu-toggle");
//    const sidebar = document.getElementById("sidebar");
//
//    menuToggle.addEventListener("click", () => {
//         sidebar.classList.toggle("show");
//    });


    // App State
    let currentUser = null;
    let currentChatId = null;
    let chats = {};
    let isSidebarOpen = true;

    // Initialize the app
    function init() {
        // Check if user is logged in (in a real app, this would check localStorage or cookies)
        const storedUser = localStorage.getItem('currentUser');
        if (storedUser) {
            currentUser = JSON.parse(storedUser);
            showApp();
        } else {
            showLoginModal();
        }

        // Load chats from localStorage
        const storedChats = localStorage.getItem('chats');
        if (storedChats) {
            chats = JSON.parse(storedChats);
            renderChatHistory();
        } else {
            createNewChat();
        }

        // Set up event listeners
        setupEventListeners();
    }

    // Set up all event listeners
    function setupEventListeners() {
        // Auth modal toggles
        showRegisterLink.addEventListener('click', (e) => {
            e.preventDefault();
            loginModal.style.display = 'none';
            registerModal.style.display = 'block';
        });

        showLoginLink.addEventListener('click', (e) => {
            e.preventDefault();
            registerModal.style.display = 'none';
            loginModal.style.display = 'block';
        });

        // Form submissions
        loginForm.addEventListener('submit', handleLogin);
        registerForm.addEventListener('submit', handleRegister);

        // Chat functionality
        newChatBtn.addEventListener('click', createNewChat);
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // File handling
        fileBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileUpload);

        // UI controls
        menuToggle.addEventListener('click', toggleSidebar);
        modelSelector.addEventListener('change', updateModel);
        
        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
        });
    }

    // Auth handlers
    function handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        
        // In a real app, this would be an API call
        if (email && password) {
            // Mock login - in a real app, verify credentials with server
            currentUser = {
                name: email.split('@')[0],
                email: email
            };
            
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            showApp();
            showToast('Login successful!');
        } else {
            document.getElementById('login-error').textContent = 'Please enter both email and password';
        }
    }

    function handleRegister(e) {
        e.preventDefault();
        const name = document.getElementById('register-name').value;
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;
        
        if (name && email && password) {
            // Mock registration - in a real app, this would create a user on the server
            currentUser = {
                name: name,
                email: email
            };
            
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            showApp();
            showToast('Account created successfully!');
        } else {
            document.getElementById('register-error').textContent = 'Please fill all fields';
        }
    }

    // Chat functionality
    function createNewChat() {
        const chatId = Date.now().toString();
        currentChatId = chatId;
        
        chats[chatId] = {
            id: chatId,
            title: 'New Chat',
            model: modelSelector.value,
            messages: [],
            createdAt: new Date().toISOString()
        };
        
        saveChats();
        renderChatHistory();
        renderChat();
        
        // Focus the input
        setTimeout(() => userInput.focus(), 100);
    }

    function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return;
        
        // Add user message to chat
        addMessageToChat('user', messageText);
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Simulate AI response after a delay
        setTimeout(() => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Generate AI response
            const aiResponse = generateAIResponse(messageText);
            addMessageToChat('bot', aiResponse);
            
            // Update chat title if it's the first exchange
            if (chats[currentChatId].messages.length === 2) {
                updateChatTitle(messageText);
            }
        }, 1000 + Math.random() * 2000); // Random delay between 1-3 seconds
    }

    function addMessageToChat(role, content) {
        if (!chats[currentChatId]) return;
        
        const message = {
            id: Date.now().toString(),
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        };
        
        chats[currentChatId].messages.push(message);
        saveChats();
        renderChat();
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function generateAIResponse(response) {
        // This is a mock response generator
        // In a real app, this would call an AI API
        
        const responses = [
            `I understand you're asking about "${response}". That's an interesting topic.`,
            `Regarding "${response}", I can provide some information on that.`,
            `I've analyzed your question about "${response}" and here's what I found.`,
            `Thanks for your question! About "${response}", here's what I know.`,
            `"${response}" is a great question. Let me explain what I understand about it.`
        ];
        
        const followUps = [
            "\n\nIs there anything specific you'd like to know more about?",
            "\n\nWould you like me to elaborate on any particular aspect?",
            "\n\nLet me know if you need more details.",
            "\n\nI can provide more examples if you're interested.",
            "\n\nFeel free to ask follow-up questions."
        ];
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        const randomFollowUp = followUps[Math.floor(Math.random() * followUps.length)];
        
        return randomResponse + randomFollowUp;
    }

    function updateChatTitle(firstMessage) {
        // Generate a title based on the first message
        let title = firstMessage.length > 30 
            ? firstMessage.substring(0, 30) + '...' 
            : firstMessage;
            
        chats[currentChatId].title = title;
        chatTitle.textContent = title;
        saveChats();
        renderChatHistory();
    }

    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message-container';
        typingIndicator.id = 'typing-indicator';
        typingIndicator.innerHTML = `
            <div class="message bot-message">
                <div class="message-content">
                    <div class="message-bubble typing">
                        <div class="typing-dots">
                            <div class="dot"></div>
                            <div class="dot"></div>
                            <div class="dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        chatContainer.appendChild(typingIndicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        showToast(`File "${file.name}" uploaded`);
        // In a real app, you would process the file here
        // For now, we'll just add a message about the file
        addMessageToChat('user', `[Attached file: ${file.name}]`);
        
        // Clear the file input
        fileInput.value = '';
    }

    // UI rendering
    function renderChat() {
        if (!currentChatId || !chats[currentChatId]) return;
        
        chatContainer.innerHTML = '';
        
        const chat = chats[currentChatId];
        chatTitle.textContent = chat.title;
        
        chat.messages.forEach(message => {
            const messageElement = document.createElement('div');
            messageElement.className = 'message-container';
            
            if (message.role === 'user') {
                messageElement.innerHTML = `
                    <div class="message user-message">
                        <div class="message-content">
                            <div class="message-bubble">
                                ${message.content}
                            </div>
                        </div>
                    </div>
                `;
            } else {
                messageElement.innerHTML = `
                    <div class="message bot-message">
                        <div class="message-content">
                            <div class="message-bubble">
                                <div class="message-actions">
                                    <button class="message-action" title="Copy">
                                        <i class="fas fa-copy"></i>
                                    </button>
                                </div>
                                ${message.content}
                            </div>
                        </div>
                    </div>
                `;
                
                // Add copy functionality
                const copyBtn = messageElement.querySelector('.message-action');
                copyBtn.addEventListener('click', () => {
                    navigator.clipboard.writeText(message.content);
                    showToast('Message copied to clipboard!');
                });
            }
            
            chatContainer.appendChild(messageElement);
        });
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function renderChatHistory() {
        chatHistory.innerHTML = '';
        
        // Sort chats by creation date (newest first)
        const sortedChats = Object.values(chats).sort((a, b) => 
            new Date(b.createdAt) - new Date(a.createdAt));
        
        sortedChats.forEach(chat => {
            const chatElement = document.createElement('div');
            chatElement.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
            chatElement.innerHTML = `
                <i class="fas fa-comment-alt"></i>
                <span class="chat-title">${chat.title}</span>
            `;
            
            chatElement.addEventListener('click', () => {
                currentChatId = chat.id;
                renderChat();
                renderChatHistory();
                
                if (window.innerWidth < 768) {
                    toggleSidebar();
                }
            });
            
            chatHistory.appendChild(chatElement);
        });
    }

    function updateModel() {
        if (currentChatId && chats[currentChatId]) {
            chats[currentChatId].model = modelSelector.value;
            saveChats();
            showToast(`Model changed to ${modelSelector.options[modelSelector.selectedIndex].text}`);
        }
    }

    // UI helpers
    function showLoginModal() {
        loginModal.style.display = 'block';
        registerModal.style.display = 'none';
        appContainer.style.display = 'none';
    }

    function showApp() {
        loginModal.style.display = 'none';
        registerModal.style.display = 'none';
        appContainer.style.display = 'flex';
        
        // Update user info
        if (currentUser) {
            userNameElement.textContent = currentUser.name;
            userEmailElement.textContent = currentUser.email;
            
            // Generate avatar from initials
            const initials = currentUser.name.split(' ').map(n => n[0]).join('').toUpperCase();
            userAvatar.textContent = initials;
            userAvatar.style.backgroundColor = stringToColor(currentUser.email);
        }
    }

    function toggleSidebar() {
        isSidebarOpen = !isSidebarOpen;
        sidebar.style.transform = isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)';
        menuToggle.innerHTML = isSidebarOpen ? '<i class="fas fa-bars"></i>' : '<i class="fas fa-times"></i>';
    }

    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    }

    // Utility functions
    function saveChats() {
        localStorage.setItem('chats', JSON.stringify(chats));
    }

    function stringToColor(str) {
        // Generate a color from a string (for avatar backgrounds)
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        
        let color = '#';
        for (let i = 0; i < 3; i++) {
            const value = (hash >> (i * 8)) & 0xFF;
            color += ('00' + value.toString(16)).substr(-2);
        }
        
        return color;
    }

    // Initialize the app
    init();
});