document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    const copyButton = document.getElementById('copyButton');
    const chatHistory = document.getElementById('chatHistory');
    
    // إرسال الرسالة عند الضغط على Enter
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // إرسال الرسالة عند النقر على الزر
    sendButton.addEventListener('click', sendMessage);
    
    // مسح المحادثة
    clearButton.addEventListener('click', function() {
        chatHistory.innerHTML = `
            <div class="welcome-message">
                <p>مرحباً! أنا سعد.AI، كيف يمكنني مساعدتك اليوم؟</p>
            </div>
        `;
    });
    
    // نسخ الرد الأخير
    copyButton.addEventListener('click', function() {
        const botMessages = document.querySelectorAll('.bot-message');
        if (botMessages.length > 0) {
            const lastBotMessage = botMessages[botMessages.length - 1];
            navigator.clipboard.writeText(lastBotMessage.textContent.trim())
                .then(() => {
                    alert('تم نسخ الرد بنجاح!');
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                });
        } else {
            alert('لا يوجد رد لنسخه!');
        }
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        addMessage(message, 'user');
        userInput.value = '';
        
        const loadingId = showLoading();
        
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            removeLoading(loadingId);
            if (data.response) {
                addMessage(data.response, 'bot');
            } else {
                addMessage('حدث خطأ في معالجة طلبك.', 'bot');
            }
        })
        .catch(error => {
            removeLoading(loadingId);
            addMessage('فشل الاتصال بالخادم. يرجى المحاولة لاحقًا.', 'bot');
            console.error('Error:', error);
        });
    }
    
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <p>${text}</p>
            <span class="message-time">${timeString}</span>
        `;
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    function showLoading() {
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.classList.add('message', 'bot-message');
        loadingDiv.innerHTML = `
            <div class="loading-dots">
                <span>.</span><span>.</span><span>.</span>
            </div>
        `;
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return loadingId;
    }
    
    function removeLoading(id) {
        const loadingElement = document.getElementById(id);
        if (loadingElement) {
            loadingElement.remove();
        }
    }
});